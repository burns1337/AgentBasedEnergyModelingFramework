import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from scipy.optimize import minimize, Bounds
from pathlib import Path

from Scripts.utils import get_evaluation_scores
from userfeedback import Agent



def optimize_AC_energy_consumption(x_test_df, start_index, model, input_scaler, lookback, horizon, max_temp=21):
    if type(max_temp) in [np.ndarray, list]:
        if len(max_temp) != horizon:
            raise ValueError("Length of temp_constraint does not match horizon!")

    def exceeded_temperature_max(x):
        pred = predict_horizon_hours_from_start_index_with_given_cooling_load(x_test_df, start_index, model,
                                                                              input_scaler, lookback,
                                                                              horizon=horizon,
                                                                              cooling_load=x)
        return (-(pred - max_temp)).min()

    constraints = {'type': 'ineq',
                   'fun': exceeded_temperature_max}  # 'ineq' means that the result of the constraint func must be non negative

    func = lambda x: x.sum()  # the actual function we want to minimize is the sum of load in W

    x0 = np.zeros(horizon)  # initial guess, no av running

    bounds = Bounds(lb=0, ub=1000)

    options = {
        'maxiter': 100,
        'disp': True,
        'ftol': 1  # 1 W tolerance seems reasonable, if smaller this runs into problems finding the minima.
    }

    opres = minimize(func, x0, method='SLSQP', constraints=constraints, bounds=bounds, options=options)
    print(opres)
    return opres


def predict_horizon_hours_from_start_index_with_given_cooling_load(x_test_df, start_index, model, input_scaler, lookback,
                                                                   horizon=12, cooling_load=None):
    predictions = []
    for i in range(0, horizon):
        x = x_test_df.iloc[i + start_index].copy()

        # set cooling load values in features:
        if cooling_load is not None:
            x['THERMAL ZONE 1 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Cooling Rate [W](Hourly) '] = cooling_load[i]
            for lb in list(range(1, lookback + 1)):
                if cooling_load is not None and i - lb >= 0:
                    x['THERMAL ZONE 1 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Cooling Rate [W](Hourly) -' + str(lb) + ''] = cooling_load[i - lb]

        # set temperature predictions back into features for next prediction
        for lb in list(range(1, lookback + 1)):
            if len(predictions) >= lb:
                x['THERMAL ZONE 1:Zone Mean Air Temperature [C](Hourly)-' + str(lb) + ''] = predictions[
                    -lb]  # set the last temperature to the one we just predicted
            if cooling_load is not None and i - lb >= 0:
                x['THERMAL ZONE 1 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Cooling Rate [W](Hourly) -' + str(lb) + ''] = cooling_load[i - lb]

        x = input_scaler.transform(x.to_frame().T)
        y_pred = model.predict(x)
        predictions.append(y_pred.flatten()[0])
    return np.array(predictions)
