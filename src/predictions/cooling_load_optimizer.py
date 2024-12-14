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

# from Scripts.utils import get_evaluation_scores
from abm.userfeedback import Agent


class CoolingLoadOptimizer:
    def __init__(self):
        self.x_test_df = None
        self.y_test_df = None
        self.model = None
        self.input_scaler = None
        self.lookback = None
        self.horizon = None
        self.features = None
        self.df = None
        self.x_train_df = None
        self.x_test = None
        self.y_train_df = None
        self.y_test = None
        self.y_pred = None


    def do_optimization(self, x_test_df, start_index, model, input_scaler, lookback, horizon, max_temp=21):
        if type(max_temp) in [np.ndarray, list]:
            if len(max_temp) != horizon:
                raise ValueError("Length of temp_constraint does not match horizon!")

        def exceeded_temperature_max(x):
            pred = self.predict_horizon_hours_from_start_index_with_given_cooling_load(x_test_df, start_index, model,
                                                                                  input_scaler, lookback,
                                                                                  horizon=horizon,
                                                                                  cooling_load=x)
            # return (-(pred - max_temp)).min()
            return (-(np.array(pred) - np.array(max_temp))).min()

        constraints = {'type': 'ineq',
                       'fun': exceeded_temperature_max}  # 'ineq' means that the result of the constraint func must be # non negative

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

    def train_linear_regrssion_with_lookback(self, lookback=12):
        df = pandas.read_csv(
            Path(Path(__file__).parent.parent, 'output', 'eplusout.csv'),
            header=0, index_col=0)  # 'final_office_IdealLoad_summer_tiny_hourly_own_schedule_24_final_2.csv'
        df = df.drop(
            labels='THERMAL ZONE 1:Zone Thermal Comfort ASHRAE 55 Simple Model Summer Clothes Not Comfortable Time [hr](Hourly)',
            axis=1)
        targets = ['THERMAL ZONE 1:Zone Mean Air Temperature [C](Hourly)']
        features = df.columns.drop(targets).to_list()

        for period in list(range(1, lookback + 1)):
            df = pandas.merge(df[features + targets].shift(periods=period).rename(
                columns={x: x + str(-period) for x in features + targets}), df, on='Date/Time')
        df = df.dropna()
        features = df.columns.drop(targets).to_list()
        self.features = features
        self.df = df

        '''
        this is essentially nowcasting (not forecasting):

        the input space of the model looks something like this: 
        f1 is a single feature, f2 is another feature, and so on.
        the input vector to the model contains all of them, t is the time now, t-1 is one hour ago.
        we know all values from the last hour (t-1), but for NOW (t) we dont know the value for |f5| but all other features.

        | t-1| t  |
        | f0 | f0 |
        | f1 | f1 |
        | f2 | f2 |  -> predict -> |x| = |f5| for time t
        | f3 | f3 |
        | f4 | f4 |
        | f5 | 
        '''

        self.x_train_df, self.x_test_df, self.y_train_df, self.y_test_df = train_test_split(df[features], df[targets], test_size=0.2,
                                                                        shuffle=False)

        input_scaler = MinMaxScaler()
        self.x_train = input_scaler.fit_transform(self.x_train_df)
        self.x_test = input_scaler.transform(self.x_test_df)

        # output_scaler = MinMaxScaler()
        # y_train = output_scaler.fit_transform(y_train)
        # y_test = output_scaler.transform(y_test)

        model = LinearRegression()
        model.fit(self.x_train, self.y_train_df)
        self.model = model

        self.y_pred = model.predict(self.x_test)

        print('linear_regrssion_with_lookback')
        self.get_evaluation_scores(self.y_pred, self.y_test_df)

        return self.x_test_df, self.y_test_df, model, input_scaler, lookback


    def run_all_horizon_prediction_with_linear_regression_wo_lookback(self, x_test_df, model, input_scaler, lookback,
                                                                      horizon=12, cooling_load=None):
        all_horizon_predictions = []
        for start_index in tqdm(range(0, x_test_df.shape[0] - horizon)):
            predictions = []
            for i in range(0, horizon):
                x = x_test_df.iloc[i + start_index].copy()
                if cooling_load:
                    x['THERMAL ZONE 1 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Cooling Rate [W](Hourly) '] = \
                        cooling_load[i]

                    # x['THERMAL ZONE 1 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Cooling Rate [W](Hourly) -' + str(lb) + ''] = cooling_load[i - lb]
                for lb in list(range(1, lookback + 1)):
                    if len(predictions) >= lb:
                        x['THERMAL ZONE 1:Zone Mean Air Temperature [C](Hourly)-' + str(lb) + ''] = predictions[
                            -lb]  # set the last temperature to the one we just predicted
                    if cooling_load and i - lb >= 0:
                        x[
                            'THERMAL ZONE 1 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Cooling Rate [W](Hourly) -' + str(
                                lb) + ''] = cooling_load[i - lb]

                x = input_scaler.transform(x.to_frame().T)
                y_pred = model.predict(x)
                predictions.append(y_pred.flatten()[0])
            all_horizon_predictions.append(predictions)
        return all_horizon_predictions

    def predict_horizon_hours_from_start_index_with_given_cooling_load(self, x_test_df, start_index, model, input_scaler,
                                                                       lookback,
                                                                       horizon=12, cooling_load=None):
        predictions = []
        for i in range(0, horizon):
            x = x_test_df.iloc[i + start_index].copy()

            # set cooling load values in features:
            if cooling_load is not None:
                x['THERMAL ZONE 1 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Cooling Rate [W](Hourly) '] = \
                cooling_load[i]
                for lb in list(range(1, lookback + 1)):
                    if cooling_load is not None and i - lb >= 0:
                        x[
                            'THERMAL ZONE 1 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Cooling Rate [W](Hourly) -' + str(
                                lb) + ''] = cooling_load[i - lb]

            # set temperature predictions back into features for next prediction
            for lb in list(range(1, lookback + 1)):
                if len(predictions) >= lb:
                    x['THERMAL ZONE 1:Zone Mean Air Temperature [C](Hourly)-' + str(lb) + ''] = predictions[
                        -lb]  # set the last temperature to the one we just predicted
                if cooling_load is not None and i - lb >= 0:
                    x[
                        'THERMAL ZONE 1 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Cooling Rate [W](Hourly) -' + str(
                            lb) + ''] = cooling_load[i - lb]

            x = input_scaler.transform(x.to_frame().T)
            y_pred = model.predict(x)
            predictions.append(y_pred.flatten()[0])
        return np.array(predictions)

    def get_representative_time_windows(self, x_test_df, sequence_r2, horizon, start_time, average_cooling_load_above):
        x_for_analysis = x_test_df.copy().reset_index()
        window_indices = x_for_analysis[x_for_analysis['Date/Time'].str.contains(start_time) & (x_for_analysis[
                                                                                                    'THERMAL ZONE 1 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Cooling Rate [W](Hourly) '].rolling(
            window=horizon).mean().shift(periods=-horizon) > average_cooling_load_above)].index

        print("most suitable eval windows are: " + str(window_indices) + ' at \n' + '\n'.join(
            x_for_analysis.loc[window_indices]['Date/Time'].to_list()))
        print("with r2 scores of :\n" + '\n'.join([str(sequence_r2[i]) for i in window_indices]))

        return window_indices

    def recursive_predict_and_eval_all_time_windows(self, x_test_df, y_test_df, model, input_scaler, lookback, horizon=12):
        all_horizon_hour_predictions = self.run_all_horizon_prediction_with_linear_regression_wo_lookback(x_test_df, model,
                                                                                                     input_scaler,
                                                                                                     lookback,
                                                                                                     horizon=horizon)

        horizon_ground_truths = []
        for start_index in range(0, x_test_df.shape[0] - horizon):
            horizon_ground_truths.append(y_test_df.iloc[start_index: start_index + horizon][
                                             'THERMAL ZONE 1:Zone Mean Air Temperature [C](Hourly)'].to_list())

        self.get_evaluation_scores(all_horizon_hour_predictions, horizon_ground_truths)

        r2s = []
        for ahead in range(0, horizon):
            r2s.append(
                r2_score(np.array(horizon_ground_truths)[:, ahead], np.array(all_horizon_hour_predictions)[:, ahead]))
        plt.plot(r2s)
        plt.ylim((0.8, 1.0))
        plt.show()

        sequence_r2 = []
        for gt, pred in zip(horizon_ground_truths, all_horizon_hour_predictions):
            sequence_r2.append(r2_score(gt, pred))

        # worst 12h sequence
        print('worst 12h sequence')
        worst_index = np.array(sequence_r2).argmin()
        self.plot_window(worst_index, horizon_ground_truths, all_horizon_hour_predictions)

        # best 12h sequence
        print('best 12h sequence')
        best_index = np.array(sequence_r2).argmax()
        self.plot_window(best_index, horizon_ground_truths, all_horizon_hour_predictions)

        print('min predicted: ' + str(np.array(all_horizon_hour_predictions).flatten().min()))
        print('max predicted: ' + str(np.array(all_horizon_hour_predictions).flatten().max()))

        return x_test_df, sequence_r2, horizon_ground_truths, all_horizon_hour_predictions, model, input_scaler, lookback

    def plot_window(self, start_index, horizon_ground_truths, all_horizon_hour_predictions):
        print('plot window ' + str(start_index))
        self.get_evaluation_scores(all_horizon_hour_predictions[start_index], horizon_ground_truths[start_index])
        plt.plot(horizon_ground_truths[start_index], label="Temp Ground Truth")
        plt.plot(all_horizon_hour_predictions[start_index], label="Temp rec. Prediction")
        plt.legend()
        plt.show()


    def run_main(self):
        ###############
        # train model
        x_test_df, y_test_df, model, input_scaler, lookback = self.train_linear_regrssion_with_lookback(lookback=2)
        print(x_test_df)
        horizon = 72

        ##############
        # run recursive prediction of 12 hours on the whole test set
        x_test_df, sequence_r2, horizon_ground_truths, all_horizon_hour_predictions, model, input_scaler, lookback = self.recursive_predict_and_eval_all_time_windows(
            x_test_df, y_test_df, model, input_scaler, lookback, horizon=horizon)

        ###############
        # get representative time windows for optimization experiment
        start_time = ' 08:00:00'  # use time windows that start at 0800 (and therefore predict up until 2000)
        # only take time windows that have an average cooling load above this (so the ones where its night and no cooling is running are too trivial
        average_cooling_load_above = 50  # works well for 12 h
        average_cooling_load_above = 20  # works well for 24 h
        setpoint = 22
        window_start_indices = self.get_representative_time_windows(x_test_df, sequence_r2, horizon, start_time,
                                                               average_cooling_load_above)

        ###############
        # The selected 2 windows that fulfill the parameters above are: 84 and 108
        # print('plot best 0800 starting window with power consumption ' + str(window_start_indices[0]))
        # plot_window(window_start_indices[0], horizon_ground_truths, all_horizon_hour_predictions)

        # print('plot best 0800 starting window with power consumption ' + str(window_start_indices[1]))
        # plot_window(window_start_indices[1], horizon_ground_truths, all_horizon_hour_predictions)

        ################
        # lets use the first one as its prediction looks better
        start_index = window_start_indices[2]

        ################
        # so now we can call predict_12h_from_start_index_with_given_cooling_load, first with the original data
        prediction_with_original_cooling_load = self.predict_horizon_hours_from_start_index_with_given_cooling_load(
            x_test_df, start_index,
            model,
            input_scaler, lookback,
            horizon=horizon)
        # print(prediction_with_original_cooling_load)

        #################
        # and this is what we want to do for the optimization task, to set a cooling load and then run:
        cooling_load = list(range(0, horizon))
        prediction_with_given_cooling_load = self.predict_horizon_hours_from_start_index_with_given_cooling_load(x_test_df,
                                                                                                            start_index,
                                                                                                            model,
                                                                                                            input_scaler,
                                                                                                            lookback,
                                                                                                            horizon=horizon,
                                                                                                            cooling_load=cooling_load)
        # print(prediction_with_given_cooling_load)
        print(
            'we can see that it is predicted to be warmer when the cooling load is 1-12 instead of being somewhere around 200 as in the original')

        ###################
        # perform optimization on selected time window WITHOUT user feedback (for comparison)
        fixed_setpoint_constraint = setpoint
        opres = self.do_optimization(x_test_df, start_index, model, input_scaler, lookback, horizon,
                                max_temp=fixed_setpoint_constraint)
        print(f"Optimization result: {opres}")
        # to csv
        pandas.DataFrame(opres).to_csv('optimization_result_opres.csv')

        predictions_with_optimization = self.predict_horizon_hours_from_start_index_with_given_cooling_load(x_test_df,
                                                                                                       start_index,
                                                                                                       model,
                                                                                                       input_scaler,
                                                                                                       lookback,
                                                                                                       horizon=horizon,
                                                                                                       cooling_load=
                                                                                                       opres['x'])

        predictions_withOUT_optimization = self.predict_horizon_hours_from_start_index_with_given_cooling_load(x_test_df,
                                                                                                          start_index,
                                                                                                          model,
                                                                                                          input_scaler,
                                                                                                          lookback,
                                                                                                          horizon=horizon,
                                                                                                          cooling_load=None)

        print('sum power original = ' + str(x_test_df[
                                                'THERMAL ZONE 1 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Cooling Rate [W](Hourly) '].iloc[
                                            start_index: start_index + horizon].to_numpy().sum()))
        print('sum power optimized = ' + str(opres['x'].sum()))

        # print('predicted temp with optimization: \n' + str(predictions_with_optimization))
        plt.plot(opres['x'], label='Optimized power')
        plt.plot(x_test_df[
                     'THERMAL ZONE 1 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Cooling Rate [W](Hourly) '].iloc[
                 start_index: start_index + horizon].to_numpy(), label='Original power')
        plt.title(f'Optimization with steady setpoint constraint ({fixed_setpoint_constraint}$^\circ$C)')
        # x labels as time and date, but only every 6th hour
        xticks = np.arange(0, len(opres['x']), 6)
        xlabels = x_test_df.index[start_index: start_index + horizon][::6]
        plt.xticks(xticks, xlabels, rotation=45, ha='right')
        plt.xlabel('Time [h]')

        # tight layout and save
        plt.tight_layout()
        plt.savefig('cooling_loads_all_22_degree_setpoint_2080.png', dpi=300)
        plt.legend()
        plt.show()

        plt.plot(y_test_df['THERMAL ZONE 1:Zone Mean Air Temperature [C](Hourly)'].iloc[
                 start_index: start_index + horizon].to_numpy(), label="Temp Ground Truth")
        plt.plot(predictions_withOUT_optimization, label="Temp predictions withOUT optimization")
        plt.plot(predictions_with_optimization, label="Temp prediction with optimized cooling load")
        plt.legend()
        plt.savefig('temperatures_all_22_degree_setpoint_2080.png', dpi=300)
        plt.show()

        ###################
        # add user feedback (single agent)
        # user = Agent((22, 22), (24, 20))
        # user = Agent((21, 21), (28, 19.5))
        user = Agent((setpoint, setpoint), (28, 19.5))
        outdoor_temp_future = x_test_df['Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)'][
                              start_index: start_index + horizon]
        setpoint_feedback_future = np.array([user.setpoint_feedback(t) for t in outdoor_temp_future])
        # print(f"setpoint_feedback_future: {setpoint_feedback_future}")

        t_eval = np.linspace(18, 30, 500)
        plt.plot(t_eval, [user.setpoint_feedback(t) for t in t_eval])
        plt.xlabel('outdoor temp')
        plt.ylabel('preferred indoor temp')
        plt.title('user feedback')
        plt.show()
        ###################
        # perform optimization on selected time window WITH user feedback
        opres_fb = self.do_optimization(x_test_df, start_index, model, input_scaler, lookback, horizon,
                                   max_temp=setpoint_feedback_future)

        predictions_with_optimization_fb = self.predict_horizon_hours_from_start_index_with_given_cooling_load(x_test_df,
                                                                                                          start_index,
                                                                                                          model,
                                                                                                          input_scaler,
                                                                                                          lookback,
                                                                                                          horizon=horizon,
                                                                                                          cooling_load=
                                                                                                          opres_fb['x'])

        predictions_withOUT_optimization_fb = self.predict_horizon_hours_from_start_index_with_given_cooling_load(x_test_df,
                                                                                                             start_index,
                                                                                                             model,
                                                                                                             input_scaler,
                                                                                                             lookback,
                                                                                                             horizon=horizon,
                                                                                                             cooling_load=None)

        print('sum power original = ' + str(x_test_df[
                                                'THERMAL ZONE 1 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Cooling Rate [W](Hourly) '].iloc[
                                            start_index: start_index + horizon].to_numpy().sum()))
        print('sum power optimized feedback= ' + str(opres_fb['x'].sum()))

        print('predicted temp with optimization: \n' + str(predictions_with_optimization))
        sum_power_original = int(round(x_test_df[
                                       'THERMAL ZONE 1 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Cooling Rate [W](Hourly) '].iloc[
                                   start_index: start_index + horizon].to_numpy().sum(), 0))
        sum_power_feedback = int(round(opres_fb['x'].sum(), 0))
        sum_power_steady_21 = int(round(opres['x'].sum(), 0))

        power_reduction_feedback_vs_original = int(round((sum_power_feedback - sum_power_original) / sum_power_original * 100))
        power_reduction_steady_vs_original = int(round((sum_power_steady_21 - sum_power_original) / sum_power_original * 100))

        if power_reduction_feedback_vs_original > 0:
            power_reduction_feedback_vs_original += 100
        if power_reduction_steady_vs_original > 0:
            power_reduction_steady_vs_original += 100

        print(f"Power reduction with feedback: {power_reduction_feedback_vs_original}%")

        print('sum power optimized fixed= ' + str(opres['x'].sum()))
        print('sum power optimized feedback = ' + str(opres_fb['x'].sum()))
        # make beautiful plot figsize bigger and save it
        fig, ax = plt.subplots(figsize=(15, 9))
        original_power = x_test_df[
            'THERMAL ZONE 1 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Cooling Rate [W](Hourly) '].iloc[
        start_index: start_index + horizon].to_numpy()
        ax.plot(opres['x'], label=f'Optimized power with steady setpoint constraint ({fixed_setpoint_constraint})$^\circ$C: {sum_power_steady_21}Wh ({power_reduction_steady_vs_original}%)')
        ax.plot(opres_fb['x'], label=f'Optmized power with feedback setpoint constraint: {sum_power_feedback}Wh ({power_reduction_feedback_vs_original}%)')
        ax.plot(original_power, label=f'Original power: {sum_power_original}Wh (100%)')

        xticks = np.arange(0, len(opres['x']), 6)
        xlabels = x_test_df.index[start_index: start_index + horizon][::6]
        plt.xticks(xticks, xlabels, rotation=45, ha='right')
        plt.xlabel('Time [h]')

        # ax.fill_between(range(len(opres_fb['x'])), opres_fb['x'], original_power,
        #                 where=(opres_fb['x'] < original_power), interpolate=True, color='green', alpha=0.3,
        #                 label='Optimized feedback vs Original')
        # ax.fill_between(range(len(opres['x'])), opres['x'], original_power,
        #                 where=(opres['x'] < original_power), interpolate=True, color='blue', alpha=0.3,
        #                 label='Optimized Steady vs Original')
        #
        # ax.fill_between(range(len(opres_fb['x'])), opres_fb['x'], opres['x'],
        #                 where=(opres_fb['x'] < opres['x']), interpolate=True, color='yellow', alpha=0.3,
        #                 label='Optimized Feedback vs Optimized Steady')

        ax.set_title('Original and optimized cooling load with steady- & user feedback setpoint')
        ax.set_xlabel('Time [h]')
        ax.set_ylabel('Cooling load [W]')
        # tight layout
        plt.tight_layout()
        ax.legend()
        plt.savefig('optimization_all_22_degree_setpoint_2080.png', dpi=300)
        plt.show()

        plt.plot(y_test_df['THERMAL ZONE 1:Zone Mean Air Temperature [C](Hourly)'].iloc[
                 start_index: start_index + horizon].to_numpy(), label="Temp Ground Truth")
        plt.plot(predictions_withOUT_optimization_fb, label="Temp rec. Prediction")
        plt.plot(predictions_with_optimization_fb, label="Optimized Temp rec. Prediction")
        plt.plot(setpoint_feedback_future, label="User constraints", color='black')
        plt.legend()
        plt.show()

        ##################
        # make figures for report
        #
        # # cooling loads
        # fig, ax = plt.subplots(2, 2, figsize=(15, 6), sharey='row', sharex=True)
        # plot_hours = 14
        # measured_power = x_test_df[
        #                      'THERMAL ZONE 1 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Cooling Rate [W](Hourly) '].iloc[
        #                  start_index: start_index + plot_hours].to_numpy()
        #
        # datecol = x_test_df.index[start_index: start_index + plot_hours].to_numpy()
        # x_labels = [x.split("  ")[1] for x in datecol]
        # date = datecol[0].split("  ")[0].replace(" ", "").split("/")
        # date = "/".join([date[1], date[0], "2022"])
        # xticks = np.arange(0, len(measured_power), 2)
        #
        # lw = 1
        #
        # ax[0, 0].plot(opres['x'][:plot_hours], linewidth=lw, label='Optimized power', color='lightseagreen')
        # ax[0, 0].plot(measured_power, linewidth=lw, label='Original power', color='coral')
        # ax[0, 0].set_title(f'Optimization with fixed setpoint constraint')
        # ax[0, 0].set_ylabel("Cooling rate [W]")
        # # ax[0, 0].set_xticks(xticks)
        # # ax[0, 0].set_xticklabels([x_labels[i] for i in xticks])
        # # ax[0, 0].set_xlabel(f"Time of day (date: {date})")
        # ax[0, 0].legend()
        #
        # ax[0, 1].plot(opres_fb['x'][:plot_hours], linewidth=lw, label='Optimized power', color='lightseagreen')
        # ax[0, 1].plot(measured_power, linewidth=lw, label='Original power', color='coral')
        # ax[0, 1].set_title('Optimization with user feedback')
        # # ax[0, 1].set_xticks(xticks)
        # # ax[0, 1].set_xticklabels([x_labels[i] for i in xticks])
        # # ax[0, 1].set_xlabel(f"Time of day (date: {date})")
        # ax[0, 1].legend()
        # # fig.suptitle('Cooling power')
        # # fig.tight_layout()
        # # plt.savefig('cooling_loads.png', dpi=300)
        #
        # # Indoor temperatures
        # # fig, ax = plt.subplots(1, 2, figsize=(15, 4), sharey=True)
        # indoor_temp_measured = y_test_df['THERMAL ZONE 1:Zone Mean Air Temperature [C](Hourly)'].iloc[
        #                        start_index: start_index + plot_hours].to_numpy()
        # ax[1, 0].plot(indoor_temp_measured, linewidth=lw, label="Temp Ground Truth", color='coral')
        # ax[1, 0].plot(predictions_withOUT_optimization[:plot_hours], linewidth=lw, label="Temp rec. Prediction",
        #               color='lightseagreen')
        # ax[1, 0].plot(predictions_with_optimization[:plot_hours], linewidth=lw, label="Optimized Temp rec. Prediction",
        #               color='purple')
        # ax[1, 0].plot(np.array([fixed_setpoint_constraint] * len(measured_power)), linewidth=lw, linestyle="--",
        #               label="fixed temp constraint", color='black')
        # # ax[1, 0].set_title('Optimization with fixed setpoint constraint')
        # ax[1, 0].set_ylabel("Indoor temp [C]")
        # ax[1, 0].set_xticks(xticks)
        # ax[1, 0].set_xticklabels([x_labels[i] for i in xticks])
        # ax[1, 0].set_xlabel(f"Time of day (date: {date})")
        # ax[1, 0].legend()
        #
        # ax[1, 1].plot(indoor_temp_measured, linewidth=lw, label="Temp Ground Truth", color='coral')
        # ax[1, 1].plot(predictions_withOUT_optimization_fb[:plot_hours], linewidth=lw, label="Temp rec. Prediction",
        #               color='lightseagreen')
        # ax[1, 1].plot(predictions_with_optimization_fb[:plot_hours], linewidth=lw,
        #               label="Optimized Temp rec. Prediction", color='purple')
        # ax[1, 1].plot(setpoint_feedback_future[:plot_hours], linewidth=lw, linestyle="--", label="User constraints",
        #               color='black')
        # # ax[1, 1].set_title('Optimization with user feedback')
        # ax[1, 1].set_xticks(xticks)
        # ax[1, 1].set_xticklabels([x_labels[i] for i in xticks])
        # ax[1, 1].set_xlabel(f"Time of day (date: {date})")
        # ax[1, 1].legend()
        #
        # fig.tight_layout()
        # plt.savefig('optimization_test2_21_08.png', dpi=300)
        # plt.show()
        #
        #
        # # second representative time window (day, 22.8.2022)
        # start_index = window_start_indices[1]
        # outdoor_temp_future = x_test_df['Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)'][
        #                       start_index: start_index + horizon]
        # setpoint_feedback_future = np.array([user.setpoint_feedback(t) for t in outdoor_temp_future])
        #
        # # print(f"setpoint_feedback_future: {setpoint_feedback_future}")
        # opres = self.do_optimization(x_test_df, start_index, model, input_scaler, lookback, horizon,
        #                         max_temp=fixed_setpoint_constraint)
        # opres_fb = self.do_optimization(x_test_df, start_index, model, input_scaler, lookback, horizon,
        #                            max_temp=setpoint_feedback_future)
        # # cooling loads
        # fig, ax = plt.subplots(2, 2, figsize=(15, 6), sharey='row', sharex=True)
        # plot_hours = 14
        # measured_power = x_test_df[
        #                      'THERMAL ZONE 1 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Cooling Rate [W](Hourly) '].iloc[
        #                  start_index: start_index + plot_hours].to_numpy()
        #
        # datecol = x_test_df.index[start_index: start_index + plot_hours].to_numpy()
        # x_labels = [x.split("  ")[1] for x in datecol]
        # date = datecol[0].split("  ")[0].replace(" ", "").split("/")
        # date = "/".join([date[1], date[0], "2022"])
        # xticks = np.arange(0, len(measured_power), 2)
        #
        # lw = 1
        #
        # ax[0, 0].plot(opres['x'][:plot_hours], linewidth=lw, label='Optimized power', color='lightseagreen')
        # ax[0, 0].plot(measured_power, linewidth=lw, label='Original power', color='coral')
        #
        # ax[0, 0].set_title(f'Optimization with fixed setpoint constraint')
        # ax[0, 0].set_ylabel("Cooling rate [W]")
        # # ax[0, 0].set_xticks(xticks)
        # # ax[0, 0].set_xticklabels([x_labels[i] for i in xticks])
        # # ax[0, 0].set_xlabel(f"Time of day (date: {date})")
        # ax[0, 0].legend()
        #
        # ax[0, 1].plot(opres_fb['x'][:plot_hours], linewidth=lw, label='Optimized power', color='lightseagreen')
        # ax[0, 1].plot(measured_power, linewidth=lw, label='Original power', color='coral')
        # ax[0, 1].set_title('Optimization with user feedback')
        # # ax[0, 1].set_xticks(xticks)
        # # ax[0, 1].set_xticklabels([x_labels[i] for i in xticks])
        # # ax[0, 1].set_xlabel(f"Time of day (date: {date})")
        # ax[0, 1].legend()
        # # fig.suptitle('Cooling power')
        # # fig.tight_layout()
        # # plt.savefig('cooling_loads.png', dpi=300)
        #
        # # Indoor temperatures
        # # fig, ax = plt.subplots(1, 2, figsize=(15, 4), sharey=True)
        # indoor_temp_measured = y_test_df['THERMAL ZONE 1:Zone Mean Air Temperature [C](Hourly)'].iloc[
        #                        start_index: start_index + plot_hours].to_numpy()
        # ax[1, 0].plot(indoor_temp_measured, linewidth=lw, label="Temp Ground Truth", color='coral')
        # ax[1, 0].plot(predictions_withOUT_optimization[:plot_hours], linewidth=lw, label="Temp rec. Prediction",
        #               color='lightseagreen')
        # ax[1, 0].plot(predictions_with_optimization[:plot_hours], linewidth=lw, label="Optimized Temp rec. Prediction",
        #               color='purple')
        # ax[1, 0].plot(np.array([fixed_setpoint_constraint] * len(measured_power)), linewidth=lw, linestyle="--",
        #               label="fixed temp constraint", color='black')
        # # ax[1, 0].set_title('Optimization with fixed setpoint constraint')
        # ax[1, 0].set_ylabel("Indoor temp [C]")
        # ax[1, 0].set_xticks(xticks)
        # ax[1, 0].set_xticklabels([x_labels[i] for i in xticks])
        # ax[1, 0].set_xlabel(f"Time of day (date: {date})")
        # ax[1, 0].legend()
        #
        # ax[1, 1].plot(indoor_temp_measured, linewidth=lw, label="Temp Ground Truth", color='coral')
        # ax[1, 1].plot(predictions_withOUT_optimization_fb[:plot_hours], linewidth=lw, label="Temp rec. Prediction",
        #               color='lightseagreen')
        # ax[1, 1].plot(predictions_with_optimization_fb[:plot_hours], linewidth=lw,
        #               label="Optimized Temp rec. Prediction", color='purple')
        # ax[1, 1].plot(setpoint_feedback_future[:plot_hours], linewidth=lw, linestyle="--", label="User constraints",
        #               color='black')
        # # ax[1, 1].set_title('Optimization with user feedback')
        # ax[1, 1].set_xticks(xticks)
        # ax[1, 1].set_xticklabels([x_labels[i] for i in xticks])
        # ax[1, 1].set_xlabel(f"Time of day (date: {date})")
        # ax[1, 1].legend()
        #
        # fig.tight_layout()
        # plt.savefig('optimization_test_22_08.png', dpi=300)
        # plt.show()
        #
        #
        # # third representative time window (day, 22.8.2022)
        # start_index = window_start_indices[2]
        # outdoor_temp_future = x_test_df['Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)'][
        #                       start_index: start_index + horizon]
        # setpoint_feedback_future = np.array([user.setpoint_feedback(t) for t in outdoor_temp_future])
        #
        # # print(f"setpoint_feedback_future: {setpoint_feedback_future}")
        # opres = self.do_optimization(x_test_df, start_index, model, input_scaler, lookback, horizon,
        #                              max_temp=fixed_setpoint_constraint)
        # opres_fb = self.do_optimization(x_test_df, start_index, model, input_scaler, lookback, horizon,
        #                                 max_temp=setpoint_feedback_future)
        # # cooling loads
        # fig, ax = plt.subplots(2, 2, figsize=(15, 6), sharey='row', sharex=True)
        # plot_hours = 14
        # measured_power = x_test_df[
        #                      'THERMAL ZONE 1 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Cooling Rate [W](Hourly) '].iloc[
        #                  start_index: start_index + plot_hours].to_numpy()
        #
        # datecol = x_test_df.index[start_index: start_index + plot_hours].to_numpy()
        # x_labels = [x.split("  ")[1] for x in datecol]
        # date = datecol[0].split("  ")[0].replace(" ", "").split("/")
        # date = "/".join([date[1], date[0], "2022"])
        # xticks = np.arange(0, len(measured_power), 2)
        #
        # lw = 1
        #
        # ax[0, 0].plot(opres['x'][:plot_hours], linewidth=lw, label='Optimized power', color='lightseagreen')
        # ax[0, 0].plot(measured_power, linewidth=lw, label='Original power', color='coral')
        # ax[0, 0].set_title(f'Optimization with fixed setpoint constraint')
        # ax[0, 0].set_ylabel("Cooling rate [W]")
        # # ax[0, 0].set_xticks(xticks)
        # # ax[0, 0].set_xticklabels([x_labels[i] for i in xticks])
        # # ax[0, 0].set_xlabel(f"Time of day (date: {date})")
        # ax[0, 0].legend()
        #
        # ax[0, 1].plot(opres_fb['x'][:plot_hours], linewidth=lw, label='Optimized power', color='lightseagreen')
        # ax[0, 1].plot(measured_power, linewidth=lw, label='Original power', color='coral')
        #
        # #TODO add sum power to plots!
        # print('sum power optimized fixed= ' + str(opres['x'].sum()))
        # print('sum power optimized feedback = ' + str(opres_fb['x'].sum()))
        #
        # ax[0, 1].set_title('Optimization with user feedback')
        # # ax[0, 1].set_xticks(xticks)
        # # ax[0, 1].set_xticklabels([x_labels[i] for i in xticks])
        # # ax[0, 1].set_xlabel(f"Time of day (date: {date})")
        # ax[0, 1].legend()
        # # fig.suptitle('Cooling power')
        # # fig.tight_layout()
        # # plt.savefig('cooling_loads.png', dpi=300)
        #
        # # Indoor temperatures
        # # fig, ax = plt.subplots(1, 2, figsize=(15, 4), sharey=True)
        # indoor_temp_measured = y_test_df['THERMAL ZONE 1:Zone Mean Air Temperature [C](Hourly)'].iloc[
        #                        start_index: start_index + plot_hours].to_numpy()
        # ax[1, 0].plot(indoor_temp_measured, linewidth=lw, label="Temp Ground Truth", color='coral')
        # ax[1, 0].plot(predictions_withOUT_optimization[:plot_hours], linewidth=lw, label="Temp rec. Prediction",
        #               color='lightseagreen')
        # ax[1, 0].plot(predictions_with_optimization[:plot_hours], linewidth=lw, label="Optimized Temp rec. Prediction",
        #               color='purple')
        # ax[1, 0].plot(np.array([fixed_setpoint_constraint] * len(measured_power)), linewidth=lw, linestyle="--",
        #               label="fixed temp constraint", color='black')
        # # ax[1, 0].set_title('Optimization with fixed setpoint constraint')
        # ax[1, 0].set_ylabel("Indoor temp [C]")
        # ax[1, 0].set_xticks(xticks)
        # ax[1, 0].set_xticklabels([x_labels[i] for i in xticks])
        # ax[1, 0].set_xlabel(f"Time of day (date: {date})")
        # ax[1, 0].legend()
        #
        # ax[1, 1].plot(indoor_temp_measured, linewidth=lw, label="Temp Ground Truth", color='coral')
        # ax[1, 1].plot(predictions_withOUT_optimization_fb[:plot_hours], linewidth=lw, label="Temp rec. Prediction",
        #               color='lightseagreen')
        # ax[1, 1].plot(predictions_with_optimization_fb[:plot_hours], linewidth=lw,
        #               label="Optimized Temp rec. Prediction", color='purple')
        # ax[1, 1].plot(setpoint_feedback_future[:plot_hours], linewidth=lw, linestyle="--", label="User constraints",
        #               color='black')
        # # ax[1, 1].set_title('Optimization with user feedback')
        # ax[1, 1].set_xticks(xticks)
        # ax[1, 1].set_xticklabels([x_labels[i] for i in xticks])
        # ax[1, 1].set_xlabel(f"Time of day (date: {date})")
        # ax[1, 1].legend()
        #
        # fig.tight_layout()
        # plt.savefig('optimization_test_23_08.png', dpi=300)
        # plt.show()

    def optimize_AC_energy_consumption(self, x_test_df, start_index, model, input_scaler, lookback, horizon, max_temp=21):
        if type(max_temp) in [np.ndarray, list]:
            if len(max_temp) != horizon:
                raise ValueError("Length of temp_constraint does not match horizon!")

        def exceeded_temperature_max(x):
            pred = self.predict_horizon_hours_from_start_index_with_given_cooling_load(x_test_df, start_index, model,
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


    def get_evaluation_scores(self, y_pred, y_test):
        # print('mse: ' + str(np.mean((y_test - y_pred) ** 2)))
        # print('mae: ' + str(np.mean(np.abs(y_test - y_pred))))
        r2_score_value = r2_score(y_test, y_pred)
        mse = np.mean((np.array(y_test) - np.array(y_pred)) ** 2)
        mae = np.mean(np.abs(np.array(y_test) - np.array(y_pred)))
        print('r2 score: ' + str(r2_score_value))
        print('mse: ' + str(mse))
        print('mae: ' + str(mae))
        return r2_score_value, mse, mae


    def optimization_on_selected_time_window_WITHOUT_user_feedback(self, x_test_df, start_index, model, input_scaler, lookback, horizon, fixed_setpoint_constraint=21):
        # perform optimization on selected time window WITHOUT user feedback (for comparison)
        fixed_setpoint_constraint = 21
        opres = self.do_optimization(x_test_df, start_index, model, input_scaler, lookback, horizon,
                                max_temp=fixed_setpoint_constraint)

        predictions_with_optimization = self.predict_horizon_hours_from_start_index_with_given_cooling_load(x_test_df, start_index,
                                                                                                       model,
                                                                                                       input_scaler, lookback,
                                                                                                       horizon=horizon,
                                                                                                       cooling_load=opres['x'])

        predictions_withOUT_optimization = self.predict_horizon_hours_from_start_index_with_given_cooling_load(x_test_df,
                                                                                                          start_index, model,
                                                                                                          input_scaler,
                                                                                                          lookback,
                                                                                                          horizon=horizon,
                                                                                                          cooling_load=None)

        print('sum power original = ' + str(
            x_test_df['THERMAL ZONE 1 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Cooling Rate [W](Hourly) '].iloc[
            start_index: start_index + horizon].to_numpy().sum()))
        print('sum power optimized = ' + str(opres['x'].sum()))
        # compare the predictions in a plot with the original power consumption
        plt.plot(opres['x'], label='Optimized power')
        plt.plot(x_test_df['THERMAL ZONE 1 IDEAL LOADS AIR SYSTEM:Zone Ideal Loads Zone Total Cooling Rate [W](Hourly) '].iloc[
                    start_index: start_index + horizon].to_numpy(), label='Original power')
        plt.plot(predictions_with_optimization, label='Optimized Temp rec. Prediction (predictions_with_optimization)')
        plt.plot(predictions_withOUT_optimization, label='Temp rec. Prediction (predictions_withOUT_optimization)')
        plt.title(f'Optimization with steady setpoint constraint ({fixed_setpoint_constraint}) $^\circ$ C')
        plt.legend()
        plt.show()



    def load_data(self, path):
        self.x_test_df = pandas.read_csv(path)
        return self.x_test_df

    def get_train_test_data(self, lookback=2):
        x_train_df = self.x_test_df.iloc[:-lookback]
        y_train_df = self.x_test_df['THERMAL ZONE 1:Zone Mean Air Temperature [C](Hourly)'].iloc[lookback:]
        x_test_df = self.x_test_df.iloc[-lookback:]
        y_test_df = self.x_test_df['THERMAL ZONE 1:Zone Mean Air Temperature [C](Hourly)'].iloc[-lookback:]
        return x_train_df, y_train_df, x_test_df, y_test_df

    def train_model_default(self, x_train_df, y_train_df):
        input_scaler = MinMaxScaler()
        x_train_scaled = input_scaler.fit_transform(x_train_df)
        model = LinearRegression()
        model.fit(x_train_scaled, y_train_df)
        return model, input_scaler

    def train_default_linear_regrssion_with_lookback(self, lookback=2):
        x_train_df, y_train_df, x_test_df, y_test_df = self.get_train_test_data(lookback=lookback)
        model, input_scaler = self.train_model_default(x_train_df, y_train_df)
        return x_test_df, y_test_df, model, input_scaler, lookback
