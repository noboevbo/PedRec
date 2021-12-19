from dataclasses import dataclass


@dataclass()
class OrientationValidationResult(object):
    angle_error_phi: float
    angle_error_theta: float
    spherical_distance: float


@dataclass()
class OrientationValidationResults(object):
    body: OrientationValidationResult
    head: OrientationValidationResult
    
    
@dataclass()
class FullOrientationValidationResult(object):
    angle_error_theta_5: float
    angle_error_theta_15: float
    angle_error_theta_22_5: float
    angle_error_theta_30: float
    angle_error_theta_45: float
    angle_error_theta_mean: float
    angle_error_theta_std: float
    angle_error_phi_5: float
    angle_error_phi_15: float
    angle_error_phi_22_5: float
    angle_error_phi_30: float
    angle_error_phi_45: float
    angle_error_phi_mean: float
    angle_error_phi_std: float
    spherical_distance_mean: float


@dataclass()
class FullOrientationValidationResults(object):
    body: OrientationValidationResult
    head: OrientationValidationResult