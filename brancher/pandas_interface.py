import pandas as pd
import numpy as np


def reformat_value(value):
    if np.prod(value.shape) == 1:
        return float(value.data)
    elif value.shape[0] == 1:
        return value.data[0, :]
    else:
        return value.data


def reformat_sample_to_pandas(sample, number_samples): #TODO: Work in progress
    data = [[reformat_value(value[index, :, :])
             for index in range(number_samples)]
            for variable, value in sample.items()]
    index = [key.name for key in sample.keys()]
    column = range(number_samples)
    return pd.DataFrame(data, index=index, columns=column)


def reformat_model_summary(summary_data, var_names, feature_list):
    return pd.DataFrame(summary_data, index=var_names, columns=feature_list)
