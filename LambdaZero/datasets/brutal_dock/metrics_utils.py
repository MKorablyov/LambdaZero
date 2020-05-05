import logging

import numpy as np


def get_prediction_statistics(list_actuals: np.array, list_predicted: np.array):
    list_absolute_errors = np.abs(list_actuals-list_predicted)
    mean_absolute_error = np.mean(list_absolute_errors)
    std_absolute_error = np.std(list_absolute_errors)

    info = f"Results [real scale]: " \
           f"mean validation values : {np.mean(list_actuals):5f}, " \
           f"std on validation values : {np.std(list_actuals):5f}, " \
           f"mean absolute error : {mean_absolute_error:5f}, " \
           f"std absolute error : {std_absolute_error:5f}."
    logging.info(info)
    return mean_absolute_error, std_absolute_error
