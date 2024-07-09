
import numpy as np
import pandas as pd
from scipy import stats, interpolate
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
from scipy.spatial import cKDTree
import copy


def filter_mean_drift(df_offsets, trialid, orientations= ["y"], 
                        x_sigma=1, y_sigma=1, t_sigma=3, filter="gaussian",
                        n_trials_before=10, n_trials_after=10):
    
    
    trialid_int = int(trialid)
    min_trialid = trialid_int - n_trials_before
    max_trialid = trialid_int + n_trials_after

    df_offsets_sub = {k: v for k, v in df_offsets.items() if min_trialid <= int(k) <= max_trialid}
    
    df = copy.deepcopy(df_offsets[trialid])

    for orientation in orientations:

        nrows = df_offsets[trialid]["point_pos_x"].nunique()
        ncols  = df_offsets[trialid]["point_pos_y"].nunique()
        num_slices = len(df_offsets_sub)
        offset_matrix = np.zeros((ncols, nrows, num_slices))

        # fill the matrix
        t = 0
        t_trialid_map = {}
        for trialid, df in df_offsets_sub.items():
            t_trialid_map[trialid] = t
            for i_row, row in df.iterrows():

                x_index = i_row % nrows  #wrap x_index correctly across columns
                y_index = i_row // ncols  #increment y_index after every 'ncols' iterations

                if y_index >= nrows:
                    print(f"Warning: y_index ({y_index}) out of bounds for trialid {trialid}")
                    continue

                offset_matrix[y_index, x_index, t] = row[f'offset_{orientation}']
                
            t += 1
        
        # filter the offset matrix
        if filter == "gaussian":
            filtered_offset_matrix = gaussian_filter(offset_matrix, sigma=(y_sigma, x_sigma, t_sigma), mode='reflect')
        elif filter == "median":
            #median filter along the t-dimension only
            filtered_offset_matrix = median_filter(offset_matrix, size=(1, 1, int(t_sigma)), mode='reflect')
        elif filter == "mean":
            #mean (uniform) filter along the t-dimension only
            filtered_offset_matrix = uniform_filter(offset_matrix, size=(1, 1, int(t_sigma)), mode='reflect')

        # fill in the changed values
        t = t_trialid_map[trialid]
        i_row = 0  
        for y_index in range(ncols):
            for x_index in range(nrows):
                if i_row < len(df):
                    df.at[i_row, f'offset_{orientation}'] = filtered_offset_matrix[y_index, x_index, t]
                i_row += 1

    return df

# def fill_with_nearest_2d(arr):
#     nrows, ncols = arr.shape
#     #forward fill
#     for i in range(ncols):
#         if np.all(np.isnan(arr[:, i])): 
#             for j in range(i + 1, ncols):
#                 if not np.all(np.isnan(arr[:, j])):
#                     arr[:, i] = arr[:, j] 
#                     break
#     # backward
#     for i in range(ncols - 1, -1, -1):
#         if np.all(np.isnan(arr[:, i])): 
#             for j in range(i - 1, -1, -1):
#                 if not np.all(np.isnan(arr[:, j])):
#                     arr[:, i] = arr[:, j] 
#                     break
#     # interpolate in case of nans
#     for i in range(ncols):
#         mask = np.isnan(arr[:, i])
#         if np.any(mask): 
#             valid_mask = ~mask
#             if valid_mask.any(): 
#                 arr[mask, i] = np.interp(np.flatnonzero(mask), np.flatnonzero(valid_mask), arr[valid_mask, i])



def fill_with_nearest_2d(array):
    mask = np.isnan(array)
    if not np.any(mask):
        return
    
    known_points = np.argwhere(~mask)
    known_values = array[~mask]
    unknown_points = np.argwhere(mask)
    
    array[mask] = interpolate.griddata(known_points, known_values, unknown_points, method='nearest')

def interpolate_drift(drift_offset, screen_res, n=50, method="cubic", fill_nan="nearest"):
    drift = {
        "on_points": {},
        "interpolated": {}
    }
    
    orig_x = np.array(drift_offset["point_pos"].apply(lambda x: x[0]))
    orig_y = np.array(drift_offset["point_pos"].apply(lambda x: x[1]))
    
    xx = np.linspace(0, screen_res[0], n)
    yy = np.linspace(0, screen_res[1], n)
    xx, yy = np.meshgrid(xx, yy)
    
    if "offset_x" not in drift_offset.columns or drift_offset["offset_x"].isna().all():
        u = np.zeros_like(orig_x)
        v = np.zeros_like(orig_y)
        u_interp = np.zeros_like(xx)
        v_interp = np.zeros_like(yy)
        points = np.column_stack((orig_x, orig_y))
        points_interp = np.column_stack((xx.flatten(), yy.flatten()))
    else:
        x = np.array(drift_offset["measured_point_pos"].apply(lambda x: x[0]))
        y = np.array(drift_offset["measured_point_pos"].apply(lambda x: x[1]))
        
        points = np.column_stack((x, y))
        points_interp = np.column_stack((xx.flatten(), yy.flatten()))
        
        u = np.array(drift_offset["offset_x"])
        v = np.array(drift_offset["offset_y"])
        
        u_interp = interpolate.griddata(points, u, (xx, yy), method=method)
        v_interp = interpolate.griddata(points, v, (xx, yy), method=method)
        
        for p in [u_interp, v_interp]:
            if fill_nan == "zero":
                np.nan_to_num(p, copy=False)
            elif fill_nan == "nearest":
                fill_with_nearest_2d(p)
            else:
                raise Exception(f"Invalid fill_nan option {fill_nan}")
        
        drift["on_points"] = {
            "points": points,
            "u": u,
            "v": v,
        }
        drift["interpolated"] = {
            "points": points_interp,
            "u": u_interp,
            "v": v_interp,
        }
    
    return drift



def calculate_drift_from_page_alignment(
    orig_fix_XY,
    corrected_fix_XY,
    fix_dur,
    nrows, ncols, screen_res, n_interp, interp_method, trim, min_fix_dur=60):

    
    valid_fix_dur_idx = np.where(fix_dur >= min_fix_dur)[0]
    
    offsets = corrected_fix_XY - orig_fix_XY
    
    orig_fix = orig_fix_XY[valid_fix_dur_idx]
    offset_fix = offsets[valid_fix_dur_idx]
    

    offsets = []

    x_bins = np.linspace(0, screen_res[0], ncols+1)
    y_bins = np.linspace(0, screen_res[1], nrows+1)
    grid_center_x = (x_bins[:-1] + x_bins[1:]) / 2
    grid_center_y = (y_bins[:-1] + y_bins[1:]) / 2

    for i_row in range(nrows):
        for i_col in range(ncols):
            x_0, x_1 = x_bins[i_col], x_bins[i_col+1]
            y_0, y_1 = y_bins[i_row], y_bins[i_row+1]
            d = {"point_pos": [grid_center_x[i_col], grid_center_y[i_row]]}

            included_fix_idx = np.where((orig_fix[:, 0] >= x_0) & (orig_fix[:, 0] < x_1) & (orig_fix[:, 1] >= y_0) & (orig_fix[:, 1] < y_1))[0]
            
            if included_fix_idx.size == 0:
                d["offset_x"] = np.nan
                d["offset_y"] = np.nan
            else:
                d["offset_x"] = stats.trim_mean(offset_fix[included_fix_idx, 0], trim)
                d["offset_y"] = stats.trim_mean(offset_fix[included_fix_idx, 1], trim)

            
            offsets.append(d)

    df_offset = pd.DataFrame(offsets)

    # interpolate NaNs linearly, then fill forward and backward to handle edge cases
    for column in df_offset.columns:
        if 'offset' in column:
            df_offset[column] = df_offset[column].interpolate(method='linear').ffill().bfill()


    df_offset[f"measured_point_pos"] = df_offset.apply(lambda row: [round(row["point_pos"][0]-row["offset_x"],2), round(row["point_pos"][1]-row["offset_y"],2)], axis=1)

        
    drift_dict = interpolate_drift(df_offset, screen_res=screen_res, n=n_interp, method=interp_method)
    return df_offset, drift_dict



def get_mean_drift_across_pages(text_drifts, settings):

    trialids = list(text_drifts.keys())
    
    # create point_pos_x /y columns
    for ti in trialids:
        text_drifts[ti]["point_pos_x"] = text_drifts[ti].apply(lambda row: row.point_pos[0], axis=1)
        text_drifts[ti]["point_pos_y"] = text_drifts[ti].apply(lambda row: row.point_pos[1], axis=1)

    central_trialid = trialids[int(len(trialids)/2):int(len(trialids)/2)+1][0]
    mean_df_offsets_eye = {ti: text_drifts[ti] for ti in trialids}
    
    filtered_df_offset = filter_mean_drift(
                    df_offsets=mean_df_offsets_eye,
                    trialid=central_trialid,
                    orientations=settings.orientations,
                    x_sigma=settings.x_sigma,
                    y_sigma=settings.y_sigma,
                    t_sigma=settings.t_sigma,
                    filter=settings.mean_filter,
                    n_trials_before=settings.n_trials_before,
                    n_trials_after=settings.n_trials_after
                    )
    drift_dict = interpolate_drift(filtered_df_offset, screen_res=settings.screen_res, n=settings.n_interp, method=settings.interp_method)

    return drift_dict, filtered_df_offset



def drift_correct_vector_field(fixation_XY, drift_dict, correct_x=True, correct_y=True, extrapolate=True):
    tmp_drift_dict = drift_dict.copy()
    points = tmp_drift_dict["interpolated"]["points"]
    
    if extrapolate:
        for direct in ["u", "v"]:
            data = tmp_drift_dict["interpolated"][direct]
            mask = data == 0
            data[mask] = np.nan
            df = pd.DataFrame(data)
            df_filled = df.ffill().bfill()
            df_filled[df_filled.isnull().all(axis=1)] = 0
            tmp_drift_dict["interpolated"][direct] = df_filled.to_numpy()
    
    u = tmp_drift_dict["interpolated"]["u"].flatten()
    v = tmp_drift_dict["interpolated"]["v"].flatten()
    
    # nearest neighbor search
    tree = cKDTree(points)
    _, indices = tree.query(fixation_XY[:, :2])
    
    x_offsets = u[indices]
    y_offsets = v[indices]
    
    np.nan_to_num(x_offsets, copy=False)
    np.nan_to_num(y_offsets, copy=False)
    
    if correct_x:
        fixation_XY[:, 0] += x_offsets
    if correct_y:
        fixation_XY[:, 1] += y_offsets
    
    return fixation_XY

