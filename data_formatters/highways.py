from sklearn.preprocessing import LabelEncoder

import data_formatters.base
import libs.utils as utils
import pandas as pd
import sklearn.preprocessing

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes

class HighwaysFormatter(GenericDataFormatter):
  """Defines and formats data for the volatility dataset.

  Attributes:
    column_definition: Defines input and data type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
  """

  _column_definition = [
      ('Global_Time', DataTypes.DATE, InputTypes.TIME),
      ('v_length', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),
      ('v_Width', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),
      ('v_Vel', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('v_Acc', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('Prec_Vel', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('FR', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('v_Class_2', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      # ('v_Class_3', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('Lane_2', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
      # ('Lane_3', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
      # ('Lane_4', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
      # ('Lane_5', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
      # ('Lane_6', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
      # ('Lane_7', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
      ('Vehicle_ID', DataTypes.CATEGORICAL, InputTypes.ID),

      ('Time_Headway', DataTypes.REAL_VALUED, InputTypes.TARGET)
  ]

  def __init__(self):
    """Initialises formatter."""

    self.identifiers = None
    self._real_scalers = None
    self._cat_scalers = None
    self._target_scaler = None
    self._num_classes_per_cat_input = None

  def split_data(self, df):

      print(" int len data: ", int(len(df)))
      cv_size = int(len(df) * 0.65)
      print("cv_size ", cv_size)
      size = int(len(df) * 0.8)

      # for train data will be collected from each country's data which index is from 0-size (80%)
      train = df.iloc[0:cv_size]
      val = df.iloc[cv_size:size]

      # for test data will be collected from each country's  data which index is from size to the end (20%)
      test = df.iloc[size:]

      print(train)
      self.set_scalers(train)

      # idk = self.transform_inputs(train)
      # print(idk)
      # print(type(idk))

      return (self.transform_inputs(data) for data in [train, val, test])

  def set_scalers(self, df):
    """Calibrates scalers using the data supplied.

    Args:
      df: Data to use to calibrate scalers.
    """
    print('Setting scalers with training data...')

    column_definitions = self.get_column_definition()
    print("column definitions",column_definitions)

    id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                   column_definitions)
    print("id_column", id_column)
    target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                       column_definitions)

    print("target_column", id_column)


    # Extract identifiers in case required
    self.identifiers = list(df[id_column].unique())

    # Format real scalers
    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    data = df[real_inputs].values
    self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data)
    self._target_scaler = sklearn.preprocessing.StandardScaler().fit(
        df[[target_column]].values)  # used for predictions

    # Format categorical scalers
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    categorical_scalers = {}
    num_classes = []
    for col in categorical_inputs:
      # Set all to str so that we don't have mixed integer/string columns
      srs = df[col].apply(str)
      categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
          srs.values)
      num_classes.append(srs.nunique())

    # Set categorical scaler outputs
    self._cat_scalers = categorical_scalers
    self._num_classes_per_cat_input = num_classes

  def transform_inputs(self, df):
    """Performs feature transformations.

    This includes both feature engineering, preprocessing and normalisation.

    Args:
      df: Data frame to transform.

    Returns:
      Transformed data frame.

    """
    output = df.copy()

    if self._real_scalers is None and self._cat_scalers is None:
      raise ValueError('Scalers have not been set!')

    column_definitions = self.get_column_definition()

    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME})
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    # Format real inputs
    output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)

    # Format categorical inputs
    for col in categorical_inputs:
      string_df = df[col].apply(str)
      output[col] = self._cat_scalers[col].transform(string_df)

    return output

  def format_predictions(self, predictions):
    """Reverts any normalisation to give predictions in original scale.

    Args:
      predictions: Dataframe of model predictions.

    Returns:
      Data frame of unnormalised predictions.
    """
    output = predictions.copy()

    column_names = predictions.columns

    for col in column_names:
      if col not in {'forecast_time', 'identifier'}:
        output[col] = self._target_scaler.inverse_transform(predictions[col])

    return output

  def format_predictions(self, predictions):
    """Reverts any normalisation to give predictions in original scale.

    Args:
      predictions: Dataframe of model predictions.

    Returns:
      Data frame of unnormalised predictions.
    """
    output = predictions.copy()

    column_names = predictions.columns

    for col in column_names:
      if col not in {'forecast_time', 'identifier'}:
        output[col] = self._target_scaler.inverse_transform(predictions[col])

    return output

  # Default params
  def get_fixed_params(self):
    """Returns fixed model parameters for experiments."""

    fixed_params = {
        'total_time_steps': 252 + 5,
        'num_encoder_steps': 252,
        'num_epochs': 100,
        'early_stopping_patience': 5,
        'multiprocessing_workers': 5,
    }

    return fixed_params

  def get_default_model_params(self):
    """Returns default optimised model parameters."""

    model_params = {
        'dropout_rate': 0.3,
        'hidden_layer_size': 160,
        'learning_rate': 0.01,
        'minibatch_size': 64,
        'max_gradient_norm': 0.01,
        'num_heads': 1,
        'stack_size': 1
    }

    return model_params