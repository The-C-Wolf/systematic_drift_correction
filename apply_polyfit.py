import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def fit_polynomial_multivariate(points, transformed, degree=2, include_interaction=True):

    poly = PolynomialFeatures(degree=degree)
    points_poly = poly.fit_transform(points)
    
    # manually add interaction term if degree < 2
    if include_interaction and degree < 2:
        interaction_term = (points[:, 0] * points[:, 1]).reshape(-1, 1)
        points_poly = np.hstack((points_poly, interaction_term))
    
    else:
        model_x = LinearRegression().fit(points_poly, transformed[:, 0])
        model_y = LinearRegression().fit(points_poly, transformed[:, 1])
    
    return model_x, model_y, poly

def apply_polynomial_warping_multivariate(points, model_x, model_y, poly):

    points_poly = poly.transform(points)
    x_transformed = model_x.predict(points_poly)
    y_transformed = model_y.predict(points_poly)
    transformed_points = np.array(np.vstack((x_transformed, y_transformed)).T)
    return transformed_points

def transform_points_polynomial_multivariate(fixation_orig_XY, trusted_fix_orig_XY, trusted_fix_correct_XY, degree=2):

    model_x, model_y, poly = fit_polynomial_multivariate(trusted_fix_orig_XY, trusted_fix_correct_XY, degree)
    transformed_points = apply_polynomial_warping_multivariate(fixation_orig_XY, model_x, model_y, poly)
    return transformed_points, (model_x, model_y, poly)



def drift_correct_polynomial_fit(
    fixation_XY, 
    trusted_fix_orig_XY,
    trusted_fix_correct_XY,
    correct_x = True,
    correct_y = True
    ):

    transformed_points_ICS, _ = transform_points_polynomial_multivariate(fixation_XY, trusted_fix_orig_XY, trusted_fix_correct_XY)
    
    if not correct_x:
        transformed_points_ICS[:,0] = fixation_XY[:,0]
    
    if not correct_y:
        transformed_points_ICS[:,1] = fixation_XY[:,1]
    
    return transformed_points_ICS        
