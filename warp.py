import numpy as np

######################################################################
# warp algorithm, adapted from jwcarr: drift algorithms
# https://github.com/jwcarr/drift/blob/master/algorithms/Python/drift_algorithms.py
# LICENSE: MIT
######################################################################


def dynamic_time_warping(sequence1, sequence2):
    # DeepSeekv2: vectorize distance calcualtion
    n1, n2 = len(sequence1), len(sequence2)
    
    # Precompute the Euclidean distances between all pairs of points using vectorized operations
    sequence1_expanded = sequence1[:, np.newaxis]
    sequence2_expanded = sequence2[np.newaxis, :]
    dist_matrix = np.sqrt(np.sum((sequence1_expanded - sequence2_expanded)**2, axis=-1))
    
    # Initialize the DTW cost matrix
    dtw_cost = np.full((n1 + 1, n2 + 1), np.inf)
    dtw_cost[0, 0] = 0

    # Fill the DTW cost matrix
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            dtw_cost[i, j] = dist_matrix[i-1, j-1] + min(dtw_cost[i-1, j], dtw_cost[i, j-1], dtw_cost[i-1, j-1])

    # Reconstruct the DTW path
    i, j = n1, n2
    dtw_path = []
    while i > 0 and j > 0:
        dtw_path.append((i-1, j-1))
        if dtw_cost[i-1, j-1] <= dtw_cost[i-1, j] and dtw_cost[i-1, j-1] <= dtw_cost[i, j-1]:
            i, j = i-1, j-1
        elif dtw_cost[i-1, j] <= dtw_cost[i-1, j-1] and dtw_cost[i-1, j] <= dtw_cost[i, j-1]:
            i -= 1
        else:
            j -= 1
    dtw_path.reverse()

    return dtw_cost[1:, 1:], dtw_path


def warp_xy(fixation_XY, word_XY, within_thresh_n_same_pos=2):

    _, dtw_path_adapt = dynamic_time_warping(fixation_XY, word_XY)

    
    dtw_path_a = {i: [] for i in range(len(fixation_XY))}
    word_path = {i: [] for i in range(len(word_XY))}
    for i_fix, j_word in dtw_path_adapt:
        dtw_path_a[i_fix].append(j_word)
        word_path[j_word].append(i_fix)
    dtw_path = list(dtw_path_a.values())
    word_path = list(word_path.values())
    within_regression_i = [i for i in word_path if len(i)>=within_thresh_n_same_pos]
    

    decide_later = []
    
    for fixation_i, words_mapped_to_fixation_i in enumerate(dtw_path):
        
        if len(words_mapped_to_fixation_i) == 1:
            fixation_XY[fixation_i, 1] = word_XY[words_mapped_to_fixation_i, 1]
            fixation_XY[fixation_i, 0] = word_XY[words_mapped_to_fixation_i, 0]
            
        else:
            decide_later.append((fixation_i, words_mapped_to_fixation_i))

            
    if len(decide_later) > 0:
        for fixation_i, words_mapped_to_fixation_i in decide_later:

            # select the line which is more frequent!
            candidate_Y = word_XY[words_mapped_to_fixation_i, 1]
            fixation_XY[fixation_i, 1] = np.mean(candidate_Y)#mode(candidate_Y)

            candidate_X= word_XY[words_mapped_to_fixation_i, 0]
            fixation_XY[fixation_i, 0] = np.mean(candidate_X)
                    
    return fixation_XY, within_regression_i