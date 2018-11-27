import pandas as pd
import numpy as np


def pandas_dict2list(dic):
    indices, values = zip(*dic.items())
    sorted_indices = np.argsort(indices)
    return np.array(values)[sorted_indices]


def pandas_frame2dict(dataframe):
    if isinstance(dataframe, pd.core.frame.DataFrame):
        return {key: pandas_dict2list(val) for key,val in dataframe.to_dict().items()}
    elif isinstance(dataframe, dict):
        return dataframe
    else:
        raise ValueError("The input should be either a dictionary or a Pandas dataframe")


def pandas_frame2value(dataframe, index):
    if isinstance(dataframe, pd.core.frame.DataFrame):
        return np.array(dataframe[index])
    else:
        return dataframe


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
    return pd.DataFrame(data, index=index, columns=column).transpose()


def reformat_model_summary(summary_data, var_names, feature_list):
    return pd.DataFrame(summary_data, index=var_names, columns=feature_list).transpose()
