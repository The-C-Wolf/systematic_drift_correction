

import os
import pandas as pd

from systematic_drift_correction import apply_text_derived_drift_correction

class Settings:
    
    screen_res = (1920, 1080)
    
    exclude_first_fix = True
    overlap_weight_factor = 10
    keep_closest_fix_within_regression = "x"
    
    min_n_lines = 7
    
    data_folder = "./data"
    stimfiles_folder = "./data/stimuli"
    
    eyes = ["l", "r"]
    
    # settings for initial alignment
    exclude_first_fix = True
    min_fix_dur = 60
    overlap_weight_factor = 10
    keep_closest_fix_within_regression = "x"
    pre_correct_first_line_offset = True
    
    # settings for drift grid
    drift_grid_n_split = 5
    n_interp = 100
    interp_method = "linear"
    trim = 0.1
    
    # settings for mean drift across trials
    orientations = ["x","y"]
    mean_filter = "median"
    n_trials_before = 20
    n_trials_after = 20
    x_sigma = 0
    y_sigma = 0
    t_sigma = 20
    
    # settings for final correction
    correct_x = True
    correct_y = True
    extrapolate = True
    
    # vis
    show_n_trials_per_subject = 0
    
    
if __name__ == "__main__":


    settings = Settings()

    fixations_fpath = os.path.join(settings.data_folder, "demo_fixations.csv")
    aois_fpath = os.path.join(settings.data_folder, "demo_aois.csv")
    
    
    all_fix = pd.read_csv(fixations_fpath)
    aois = pd.read_csv(aois_fpath)
    

    TD_corrections = []
    for subject in all_fix["subject"].unique():

        fix = all_fix[all_fix["subject"] == subject]
        fix_TD_corrected = apply_text_derived_drift_correction(fix, aois, subject, settings)
        TD_corrections.append(fix_TD_corrected)
    
    corrected_fix = pd.concat(TD_corrections)
    corrected_fix.to_csv(os.path.join(settings.data_folder, "demo_TD_corrected_fixations.csv"), index=False)
    
    
    