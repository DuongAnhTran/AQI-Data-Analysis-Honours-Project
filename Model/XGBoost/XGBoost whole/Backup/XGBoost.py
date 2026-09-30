# %% Cell 1: Imports
from doctest import OutputChecker

import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.metrics import mae
from darts.models import ARIMA, LinearRegressionModel, RandomForest, XGBModel

import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
import pyarrow as pa
import datetime as dt


# %% Cell 2: Load Data
def loadData():
    df_int = pd.read_parquet("gafanha.parquet")
    df_int["timestamp"] = pd.to_datetime(df_int["timestamp"])
    df_int = df_int.sort_values("timestamp")


    dup_check = df_int.duplicated(subset=["timestamp", "collection"], keep=False)
    print(f"Duplicate day+pollutant rows: {dup_check.sum()}")

    df = df_int.pivot_table(
        index="timestamp",
        columns="collection",
        values="value",
        aggfunc="median"
    ).sort_index()
    df = df.asfreq("h")

    df = df.reset_index()
    print(df.columns.tolist())
    #print(df[df['timestamp'].dt.year == 2017].head())
    return df

df = loadData()


# %% Cell 3: Readings per day per variable - for validating the dataset only
def checkDailyCount(df):
    df_daily_counts = df.set_index("timestamp").resample("D").count()

    # quick visual: counts per year for each variable
    df_daily_counts.groupby(df_daily_counts.index.year).mean().plot(
        kind="bar", figsize=(12, 5), title="Avg daily reading count per variable, by year"
    )
    plt.ylabel("avg readings/day")
    plt.tight_layout()
    plt.show()




# %% Cell 4: Force daily, median and fill in missing values with darts package
from darts.utils.missing_values import fill_missing_values
daily = df.set_index("timestamp").resample("D").median()
daily = daily.asfreq("D")

# interpolates NaNs values
daily.index = daily.index.tz_localize(None)
series = TimeSeries.from_dataframe(daily, freq="D")
series = fill_missing_values(series)



# %% Cell 5: set target and split
target = series["PM25"]
covariates = series.drop_columns(["PM25"])
y_train, y_temp = target.split_before(0.6)
y_val, y_test = y_temp.split_before(0.5)

x_train, x_temp = covariates.split_before(0.6)
x_val, x_test = x_temp.split_before(0.5)



# %% Cell 6: Hyperparameter search for XGBoost
import optuna
import xgboost
from darts.models import XGBModel
from darts.metrics import rmse, mape

def findParam(trial):
    """
        Testing with: predict next day, 50-400 trees, 3-10 max depth, 0.01-0.3 learning rate, 0.5-1.0 subsample, 0.5-1.0 colsample_bytree
    """
    model = XGBModel(
        lags=trial.suggest_int("lags", 7, 60),
        lags_past_covariates = trial.suggest_int("lags_past_covariates", 7, 30),
        output_chunk_length = 1,
        n_estimators = trial.suggest_int("n_estimators", 50, 400),
        max_depth = trial.suggest_int("max_depth", 3, 10),
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3),
        subsample = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
    )
    model.fit(y_train, past_covariates=x_train)
    pred = model.predict(n=len(y_val), past_covariates=x_train.append(x_val))

    trial.set_user_attr("mape", mape(y_val, pred))
    trial.set_user_attr("mae", mae(y_val, pred))
    return rmse(y_val, pred)

# Create the study
study = optuna.create_study(direction="minimize")
study.optimize(findParam, n_trials=100, show_progress_bar=True)
print(study.best_params, study.best_value)
print(f"mape: {study.best_trial.user_attrs['mape']} | mae: {study.best_trial.user_attrs['mae']} | rmse: {study.best_value}")




# %% Cell 7 checking final model, retrain on train-val and oneshot-shot eval on test
final_y_train = y_train.append(y_val)
final_x_train = x_train.append(x_val)
print(study.best_params)

best_model = XGBModel(**study.best_params, output_chunk_length=1,)
best_model.fit(final_y_train, past_covariates=final_x_train)
test_pred = best_model.predict(n=len(y_test), past_covariates=final_x_train.append(x_test))
print(f"Test RMSE: {rmse(y_test, test_pred)} || MAE: {mae(y_test, test_pred)} || MAPE: {mape(y_test, test_pred)}")
print("RMSE:", rmse(y_test, test_pred))
print("RMSE as % of mean:", rmse(y_test, test_pred) / float(y_test.mean().values()[0][0]) * 100)
rmse_fin = rmse(y_test, test_pred)
mae_fin = mae(y_test, test_pred)
mape_fin = mape(y_test, test_pred)


# %% Cell 8: Naive baseline, multi-step (matches recursive setup)
horizon = len(y_test)
last_known = y_train.append(y_val).values()[-1][0]  # last actual before test starts

naive_multistep = TimeSeries.from_times_and_values(
    y_test.time_index, np.full(horizon, last_known)
)

print("Naive multi-step RMSE:", rmse(y_test, naive_multistep))
print("Naive multi-step MAE:", mae(y_test, naive_multistep))
# %% Cell 9: Error by step-into-horizon
err = np.abs(y_test.values().flatten() - test_pred.values().flatten())
plt.plot(range(1, horizon + 1), err)
plt.xlabel("steps ahead")
plt.ylabel("abs error")
plt.title("Does error grow with recursion depth?")
plt.show()




# %% Cell 10: Check covariate coverage across test horizon
print(covariates.time_index.min(), covariates.time_index.max())
print(y_test.time_index.min(), y_test.time_index.max())
# %% Cell 11: Export, with honest reliable-horizon note
RELIABLE_HORIZON = 200

best_model.save(f"xgb_pm25_rmse{rmse_fin}_mae{mae_fin}_mape{mape_fin}.pkl")
print(f"Model saved. Reliable for ~{RELIABLE_HORIZON} days ahead recursively, "
      f"degrades/saturates beyond that per test-set error curve.")
