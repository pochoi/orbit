from .template import BaseTemplate, FullBayesianTemplate

from ..constants import lm as constants

class BaseLM(BaseTemplate):
    """Base Linear Model

    Parameters
    ----------
    regressor_col : list
        Names of regressor columns, if any
    """
    
    _data_input_mapper = constants.DataInputMapper
    _model_name = 'lm'

    def __init__(self, regressor_col=None, **kwargs):
        super().__init__(**kwargs)

        self.regressor_col = regressor_col
        self._num_of_regressors = 0
        self._regressor_col = list()

        self._response = None
        self._num_of_observations = None

    def fit(self, df):
        """Fit model to data and set extracted posterior samples"""
        estimator = self.estimator
        model_name = self._model_name

