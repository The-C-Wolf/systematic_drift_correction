# add directory to path
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from apply_polyfit import drift_correct_polynomial_fit
from warp import warp_xy


def get_word_xy(sub_aois):

    word_X = sub_aois["x"] + sub_aois["width"] / 2
    word_Y = sub_aois["y"] + sub_aois["height"] / 2
    word_XY = pd.DataFrame([word_X, word_Y]).transpose().to_numpy()
        
    return word_XY


def get_offset_above_first_line(fixation_XY, word_XY, N_first = 3):
    line_y = np.unique(word_XY[:, 1])
    general_y_offset = 0
    y_pos_sorted = np.sort(fixation_XY[:, 1])
    
    most_upper_fix_y = np.median(y_pos_sorted[N_first])
    first_line_y = np.min(line_y)
    if most_upper_fix_y < first_line_y:
        general_y_offset = first_line_y - most_upper_fix_y
    
    return general_y_offset

def get_start_end_line_aois(word_XY):
    line_y = np.unique(word_XY[:, 1])
    start_line_aois = []
    end_line_aois = []
    
    
    for y in line_y:
        aois = np.where(word_XY[:, 1]==y)[0]
        start_line_aois.append(aois[0])
        end_line_aois.append(aois[-1])
    return start_line_aois, end_line_aois

def filter_within_regression(within_regression_i,fixation_XY, word_XY, orig_fix_xy, keep_closest_fix_within_regression="x"):
    within_regression_fix_i = []
    start_line_aois, end_line_aois = get_start_end_line_aois(word_XY)
        
    for l in within_regression_i:
        x,y = fixation_XY[l[0]]
        # get aoi
        aoi_i = np.where((word_XY[:,0]==x)&(word_XY[:, 1]==y))[0][0]
        if aoi_i in start_line_aois:
            # keep only the first, collect the rest to drop
            within_regression_fix_i.extend(l[1:])
        elif aoi_i in end_line_aois:
            # keep only the last, collect the rest to drop
            within_regression_fix_i.extend(l[:-1])
        else:
            # keep only the closest, collect the rest to drop
            closest_fix_x = np.argmin([np.abs(word_XY[aoi_i, 0]-f[0]) for f in orig_fix_xy[l]])
            closest_fix_xy = np.argmin([np.abs(word_XY[aoi_i, 0]-f[0])+np.abs(word_XY[aoi_i, 1]-f[1]) for f in orig_fix_xy[l]])
            if keep_closest_fix_within_regression == "x":
                closest_fix = closest_fix_x
            else:
                closest_fix = closest_fix_xy
            keep = [idx for i, idx in enumerate(l) if i != closest_fix]
            within_regression_fix_i.extend(keep)

    return within_regression_fix_i


def get_overlapping_fixations(fixation_XY, word_XY, sub_aois, orig_fix_xy):
    overlapping_fix_i = []
    # start_line_aois, end_line_aois = get_start_end_line_aois(word_XY)
    sub_aois["x_right"] = sub_aois["x"] + sub_aois["width"]
    
    for i, (moved_fix, orig_fix) in enumerate(zip(fixation_XY, orig_fix_xy)):
        x,y = moved_fix
        # get aoi
        line_aois = np.where(word_XY[:, 1]==y)[0]
        if len(line_aois) < 2:
            continue
        first_aoi_i = line_aois[0]
        last_aoi_i = line_aois[-1]
        
        if orig_fix[0] < sub_aois.iloc[first_aoi_i]["x"]:
            overlapping_fix_i.append(i)
        elif orig_fix[0] > sub_aois.iloc[last_aoi_i]["x_right"]:
            overlapping_fix_i.append(i)
        
    return overlapping_fix_i    


def swap_crossed_fixations(fixation_XY, orig_fix_xy):

    orig_x_consec_diff = np.diff(orig_fix_xy[:, 0])
    after_x_consec_diff = np.diff(fixation_XY[:, 0])
    
    sign_changes = np.sign(orig_x_consec_diff) * np.sign(after_x_consec_diff) < 0
    sign_change_indices = np.where(sign_changes)[0]
    sign_change_index_pairs = [(i, i+1) for i in sign_change_indices]
    
    for i, j in sign_change_index_pairs:
        fixation_XY[[i, j]] = fixation_XY[[j, i]]
    return fixation_XY


def get_outlier(fixation_XY, orig_fix_xy, within_regression_fix_i, outlier_n_std):

    # detect large x jumps
    x_diff = np.abs(fixation_XY[:, 0] - orig_fix_xy[:, 0])

    # without regression idx
    indices = np.arange(x_diff.size)
    mask = ~np.isin(indices, within_regression_fix_i)
    x_diff_trimmed = x_diff[mask]
    
    
    y_diff = np.abs(fixation_XY[:, 1] - orig_fix_xy[:, 1])
    indices = np.arange(y_diff.size)
    mask = ~np.isin(indices, within_regression_fix_i)
    y_diff_trimmed = y_diff[mask]
    
    T_x_outlier = np.mean(x_diff_trimmed) + outlier_n_std * np.std(x_diff_trimmed)
    T_y_outlier = np.mean(y_diff_trimmed) + outlier_n_std * np.std(y_diff_trimmed)

    x_outlier_i = np.where(x_diff > T_x_outlier)[0]
    y_outlier_i = np.where(y_diff > T_y_outlier)[0]
    return x_outlier_i, y_outlier_i



def duplicate_entries(points, overlap_indices, overlap_weight_factor):
    """duplicate points to apply weights"""

    new_rows = []
    for i in range(points.shape[0]):
        new_rows.append(points[i])
        
        if i in overlap_indices:
            for _ in range(overlap_weight_factor):
                new_rows.append(points[i])

    return np.array(new_rows)


def auto_align_single_trial(fixation_XY, subline_aois, exclude_first_fix=True, 
               keep_closest_fix_within_regression = "x", pre_correct_first_line_offset=True, settings=None, stimfile=None):
    
    word_XY = get_word_xy(subline_aois)
    
    # make a copy before dropping anything
    orig_fix_XY_all = fixation_XY.copy()
    
    if exclude_first_fix:
        fixation_XY = fixation_XY[1:]
    
    # warp sometimes struggles with a general vertical offset, so that the first and maybe even the second fixation-line-run is above the first line
    general_y_offset = 0
    orig_fix_xy = fixation_XY.copy()
    if  pre_correct_first_line_offset:
        general_y_offset = get_offset_above_first_line(fixation_XY, word_XY, N_first = 3)
        fixation_XY[:, 1] += general_y_offset

    # # potentially apply word/aoi weights (assuming certain aois have a higher p too look at)
    # word_XY_weights = np.ones(word_XY.shape[0])
    
    fixation_XY, within_regression_i = warp_xy(fixation_XY, word_XY, within_thresh_n_same_pos=2)
    
    # within regressions lead to multiple fixations being assigned to a single AOI
    within_regression_fix_i = filter_within_regression(within_regression_i, fixation_XY, word_XY, orig_fix_xy, keep_closest_fix_within_regression)

    # small within regressions (single fixation) could stay, but could be swapped, to assume a forward reading direction
    fixation_XY = swap_crossed_fixations(fixation_XY, orig_fix_xy)
    
    x_outlier_i, y_outlier_i = get_outlier(fixation_XY, orig_fix_xy, within_regression_fix_i, outlier_n_std = 1.5)
    within_regression_fix_i = np.concatenate((within_regression_fix_i, x_outlier_i))
    between_regression_fix_i = y_outlier_i
    take_out_i = np.concatenate((within_regression_fix_i, between_regression_fix_i))
    
    # the rest is trusted
    trusted_moved = np.array(list(set(range(len(fixation_XY))).difference(take_out_i)))
    
    trusted_fix_orig_XY = orig_fix_xy[trusted_moved]
    trusted_fix_correct_XY = fixation_XY[trusted_moved]
    
    fix_XY = drift_correct_polynomial_fit(orig_fix_XY_all, trusted_fix_orig_XY, trusted_fix_correct_XY)
    
    return fix_XY