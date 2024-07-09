import pandas as pd

from align import auto_align_single_trial
from drift_grid import calculate_drift_from_page_alignment, get_mean_drift_across_pages, drift_correct_vector_field
from vis import show_trial

def apply_text_derived_drift_correction(fixations, aois, subject, settings):  

    TD_corrected = []

    for eye in settings.eyes:
        
        print(f"Processing subject {subject}, eye {eye}...")
        
        df_offsets = {}
        
        for trial_id, trial_fix in fixations.groupby("trialid"):
            
            trial_stimfile = trial_fix.stimfile.unique()[0]
        
            # drop trials with too few lines
            trial_aois = aois[aois.stimfile == trial_stimfile]
            if trial_aois.line.max() < settings.min_n_lines:
                continue

            trial_eye_fix = trial_fix[trial_fix.eye == eye]
            
            # STEP 1: run alignment via algorithm (warp)
            fixation_XY = trial_eye_fix[["orig_fix_x", "orig_fix_y"]].astype(float).to_numpy()
            orig_fix_XY = fixation_XY.copy()
            subline_aois = trial_aois[trial_aois.kind == "sub-line"]
            aligned_trial_eye_fix = auto_align_single_trial(
                fixation_XY=fixation_XY,
                subline_aois=subline_aois,
                exclude_first_fix=settings.exclude_first_fix,
                keep_closest_fix_within_regression=settings.keep_closest_fix_within_regression,
                pre_correct_first_line_offset=settings.pre_correct_first_line_offset,
                settings=settings,
                stimfile=trial_stimfile
            )
            
            # # STEP 2: now derive a drift grid from it
            df_offsets[trial_id], _ = calculate_drift_from_page_alignment(
                orig_fix_XY=orig_fix_XY,
                corrected_fix_XY=aligned_trial_eye_fix,
                fix_dur=trial_eye_fix["duration_ms"].astype(float).to_numpy(),
                nrows = settings.drift_grid_n_split,
                ncols = settings.drift_grid_n_split,
                screen_res=settings.screen_res,
                n_interp=settings.n_interp,
                interp_method=settings.interp_method,
                trim=settings.trim,
                min_fix_dur=settings.min_fix_dur,
            )


        # STEP 3: filter across pages
        mean_drift_dict, _ = get_mean_drift_across_pages( df_offsets, settings)

        # STEP 4: apply mean drift for each trial
        i_page = 1
        for trial_id, trial_fix in fixations.groupby("trialid"):
            
            trial_stimfile = trial_fix.stimfile.unique()[0]
            
            trial_eye_fix = trial_fix[trial_fix.eye == eye].copy()
            fixation_XY = trial_eye_fix[["orig_fix_x", "orig_fix_y"]].astype(float).to_numpy()
            orig_fix_XY = fixation_XY.copy()
            
            corrected_fix_XY = drift_correct_vector_field(
                fixation_XY,
                mean_drift_dict, 
                settings.correct_x,
                settings.correct_y,
                settings.extrapolate
                )
            
            trial_eye_fix.loc[:, "TD_corrected_fix_x"] = corrected_fix_XY[:, 0]
            trial_eye_fix.loc[:, "TD_corrected_fix_y"] = corrected_fix_XY[:, 1]

            TD_corrected.append(trial_eye_fix)
            
            if i_page <= settings.show_n_trials_per_subject:
                show_trial(orig_fix_XY, corrected_fix_XY, subject, trial_stimfile, eye, settings.stimfiles_folder)
                i_page += 1
    
    return pd.concat(TD_corrected)  
