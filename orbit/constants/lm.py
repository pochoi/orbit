from enum import Enum


class DataInputMapper(Enum):
    """
    mapping from object input to sampler
    """
    # ----------  Data Input ---------- #
    # observation related
    _NUM_OF_OBSERVATIONS = 'NUM_OF_OBS'
    _RESPONSE = 'RESPONSE'
    _NUM_OF_REGRESSORS = 'NUM_OF_REGRESSOR'
    _REGRESSOR_MATRIX = 'REGRESSOR_MAT'

class RegressionSamplingParameters(Enum):
    """
    regression component related parameters in posteriors sampling
    """
    REGRESSION_MEAN = 'alpha'
    REGRESSION_COEFFICIENTS = 'beta'
    RESIDUAL_SIGMA = 'obs_sigma'
    
