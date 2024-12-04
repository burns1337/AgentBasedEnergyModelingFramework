from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import requests


def get_evaluation_scores(y_pred, y_test):
    score_r2 = r2_score(y_test, y_pred)
    score_mse = mean_squared_error(y_test, y_pred)
    score_mae = mean_absolute_error(y_test, y_pred)
    score_mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f'R2-Score: {score_r2:.3f}')
    print(f'Mean Squared Error (MSE): {score_mse:.3f}')
    print(f'Mean Absolute Error (MAE): {score_mae:.3f}')
    print(f'Mean Absolute Percentage Error (MAPE): {score_mape:.3f}')


