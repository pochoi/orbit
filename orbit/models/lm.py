import pandas as pd
import numpy as np

from .template import BaseTemplate, FullBayesianTemplate
from ..exceptions import IllegalArgument, ModelException, PredictionException

from ..constants import lm as constants
from ..utils.general import is_ordered_datetime

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

    def _validate_training_df(self, df):
        df_columns = df.columns

        # validate date_col
        if self.date_col not in df_columns:
            raise ModelException("DataFrame does not contain `date_col`: {}".format(self.date_col))

        # validate ordering of time series
        date_array = pd.to_datetime(df[self.date_col]).reset_index(drop=True)
        if not is_ordered_datetime(date_array):
            raise ModelException('Datetime index must be ordered and not repeat')

        # validate response variable is in df
        if self.response_col not in df_columns:
            raise ModelException("DataFrame does not contain `response_col`: {}".format(self.response_col))

    def _set_dynamic_data_attributes(self, df):
        """Set required input based on input DataFrame, rather than at object instantiation"""
        df = df.copy()

        self._validate_training_df(df)
        self._set_training_df_meta(df)

        # a few of the following are related with training data.
        self._response = df[self.response_col].values
        self._num_of_observations = len(self._response)
        self._response_sd = np.std(self._response)


    def _set_training_df_meta(self, df):
        # Date Metadata
        # TODO: use from constants for dict key
        self._training_df_meta = {
            'date_array': pd.to_datetime(df[self.date_col]).reset_index(drop=True),
            'df_length': len(df.index),
            'training_start': df[self.date_col].iloc[0],
            'training_end': df[self.date_col].iloc[-1]
        }

    def _set_model_param_names(self):
        """Set posteriors keys to extract from sampling/optimization api"""
        self._model_param_names += [param.value for param in constants.BaseSamplingParameters]

    def fit(self, df):
        """Fit model to data and set extracted posterior samples"""
        estimator = self.estimator
        model_name = self._model_name

