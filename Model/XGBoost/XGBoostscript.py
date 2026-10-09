# %% [markdown]
# # Importing

# %%
# %% Cell 1: Imports
from doctest import OutputChecker

import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import ARIMA, LinearRegressionModel, RandomForest, XGBModel

seed = 24775456

import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
import pyarrow as pa
import datetime as dt

# %% [markdown]
# # Load Data 

# %%
# %% Cell 2: Load Data
def loadData():
    df = pd.read_parquet("gafanha.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.round("h")
    print(df.count())
    return (
        df.pivot_table(index="timestamp", columns="collection", values="value", aggfunc="median")
        .asfreq("h")
        .loc[:"2024-12-31 23:00"]
        .reset_index()
    )

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

# %%
# %% Cell 4: Force daily, median and fill in missing values with darts package
from darts.utils.missing_values import fill_missing_values
daily = df.set_index("timestamp").resample("D").median()
daily = daily.asfreq("D")

print(daily.count())


# Set up the AQI bins for each pollutant based on the standard AQI breakpoints (European standards)
aqi_bins = {
    "PM25": [-np.inf, 10, 20, 25, 50, np.inf],
    "PM10": [-np.inf, 20, 35, 50, 100, np.inf],
    "NO2":  [-np.inf, 40, 100, 200, 400, np.inf],
    "O3":   [-np.inf, 80, 120, 180, 240, np.inf],
    "SO2":  [-np.inf, 100, 200, 350, 500, np.inf],
}
sub_idx = pd.concat(
    [pd.cut(daily[p], bins=b, labels=[1, 2, 3, 4, 5]).astype(float) for p, b in aqi_bins.items()],
    axis=1,
)
daily["AQI"] = sub_idx.max(axis=1, skipna=True)
dominant = sub_idx.dropna(how="all").idxmax(axis=1)
last_valid = daily["AQI"].last_valid_index() 
print(last_valid)

# interpolates NaNs values
daily.index = daily.index.tz_localize(None)
series = TimeSeries.from_dataframe(daily, freq="D")
series = fill_missing_values(series)

# %%
print(sub_idx.notna().groupby(sub_idx.index.year).mean())
dominant.groupby(dominant.index.year).value_counts(normalize=True).unstack()

# %% [markdown]
# # Set up and prepare for training 3 XGBoost Models

# %%
# %% Cell 5: Set targets and covariates
target = series["AQI"]
covariates = series.drop_columns(["AQI"])

# %%
# %% Cell 6: generic per-period pipeline
import optuna
from darts.metrics import rmse, mae, mape

def run_period(target, covariates, model_cls, param_space, fixed_param, n_trials=500, params=None):

    # Split the data into training, validation, and test sets
    y_train, y_temp = target.split_before(0.6)
    y_val, y_test = y_temp.split_before(0.5)
    x_train, x_temp = covariates.split_before(0.6)
    x_val, x_test = x_temp.split_before(0.5)

    # Tune with Optuna only if no params were given
    if params is None:
        def findParam(trial):
            p = {name: fn(trial) for name, fn in param_space.items()}
            model = model_cls(**p, **fixed_param)
            model.fit(y_train, past_covariates=x_train)
            preds = model.historical_forecasts(
                series=y_train.append(y_val), past_covariates=x_train.append(x_val),
                start=y_val.start_time(), forecast_horizon=1, stride=1,
                retrain=False, last_points_only=True, verbose=False,
            ).map(lambda x: np.round(x))
            return rmse(y_val, preds), mae(y_val, preds), mape(y_val, preds)

        # Minimise all three metrics (RMSE, MAE, MAPE) simultaneously
        study = optuna.create_study(directions=["minimize"] * 3,
                                    sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(findParam, n_trials=n_trials, show_progress_bar=True)

        vals = np.array([t.values for t in study.best_trials])
        best = study.best_trials[np.argmin((vals / vals.min(axis=0)).sum(axis=1))]
        params = best.params

    y_fit, x_fit = y_train.append(y_val), x_train.append(x_val)
    test_pred = model_cls(**params, **fixed_param).historical_forecasts(
        series=y_fit.append(y_test), past_covariates=x_fit.append(x_test),
        start=y_test.start_time(), forecast_horizon=1, stride=1,
        retrain=True, train_length=None, last_points_only=True, verbose=False,
    ).map(lambda x: np.round(x))

    # naive = TimeSeries.from_times_and_values(
    #     y_test.time_index, target.to_series().shift(1).loc[y_test.time_index].values)

    # Fit the final model on the combined training and validation sets - for SHAP analysis and feature importance
    model = model_cls(**params, **fixed_param).fit(y_fit, past_covariates=x_fit)

    return dict(
        model=model, params=params, y=target, x=covariates, y_test=y_test, x_test=x_test, test_pred=test_pred,
        model_metrics=(rmse(y_test, test_pred), mae(y_test, test_pred), mape(y_test, test_pred)),
        # naive_metrics=(rmse(y_test, naive), mae(y_test, naive), mape(y_test, naive)),
        sizes=(len(y_train), len(y_val), len(y_test)),
    )

# %% [markdown]
# # For Each of the Periods

# %% [markdown]
# ### Config the models - run search

# %%
# %% Cell 7: XGBoost config
from darts.models import XGBModel
import joblib

xgb_fixed = dict(lags=7, lags_past_covariates=7, output_chunk_length=1,
                 tree_method="hist", n_jobs=-1, random_state=seed, objective="reg:absoluteerror")
xgb_space = dict(
    n_estimators=lambda t: t.suggest_int("n_estimators", 50, 400),
    max_depth=lambda t: t.suggest_int("max_depth", 3, 10),
    learning_rate=lambda t: t.suggest_float("learning_rate", 0.01, 0.3),
    subsample=lambda t: t.suggest_float("subsample", 0.5, 1.0),
    colsample_bytree=lambda t: t.suggest_float("colsample_bytree", 0.5, 1.0),
)

# 3 periods: pre, during, post
bounds = {"pre":    (None, "2020-02-29"),
          "during": ("2020-03-01", "2021-04-30"),
          "post":   ("2021-05-01", None)}

# %% [markdown]
# ### Refit and recreate result

# %%
results = {}
for name, (s, e) in bounds.items():
    lo = pd.Timestamp(s) if s else target.start_time()
    hi = pd.Timestamp(e) if e else target.end_time()
    results[name] = run_period(target.slice(lo, hi), covariates.slice(lo, hi), XGBModel, xgb_space, xgb_fixed)
    r = results[name]
    print(name, r["sizes"], "model", r["model_metrics"])
    r["model"].save(f"xgb_aqi_{name}.pkl")
joblib.dump(results, "results_xgb.pkl")

# %% [markdown]
# ### Build the results.pkl - run once after training

# %%
p = pd.read_csv("report_params.csv", index_col=0)
p[["n_estimators", "max_depth"]] = p[["n_estimators", "max_depth"]].astype(int)
pp = p.to_dict("index")

results = {}
for name, (s, e) in bounds.items():
    lo = pd.Timestamp(s) if s else target.start_time()
    hi = pd.Timestamp(e) if e else target.end_time()
    results[name] = run_period(target.slice(lo, hi), covariates.slice(lo, hi),
                               XGBModel, None, xgb_fixed, params=pp[name])
    print(name, results[name]["model_metrics"])
joblib.dump(results, "results_xgb.pkl")

# %% [markdown]
# # Cross Validation between models

# %% [markdown]
# ### Load the models

# %%
results = joblib.load("results_xgb.pkl")

# %%
# %% Cell 8: cross-evaluation

# Evaluate each model on each test window, and report RMSE and MAE
# Frozen models (retrain=False): diagonal differs slightly from report (retrain=True) because the model 
#   is force to not retrain to avoid diluting the training results with the test data. The diagonal is therefore slightly 
#   worse than the report, but the off-diagonal is more accurate.

def score(model, y_w):
    end = y_w.end_time()
    p = model.historical_forecasts(
        series=target.slice(target.start_time(), end),
        past_covariates=covariates.slice(covariates.start_time(), end),
        start=y_w.start_time(), forecast_horizon=1, stride=1,
        retrain=False, last_points_only=True, verbose=False,
    ).map(lambda x: np.round(x))
    return rmse(y_w, p), mae(y_w, p)

cross = pd.DataFrame(
    {test: {f"{m}-model": score(results[m]["model"], results[test]["y_test"])[0]
            for m in results}
     for test in results}
).round(3)
cross.index.name, cross.columns.name = "model", "test window (RMSE)"
print(cross)

# %% [markdown]
# # Report, SHAP and other analysis visualisation

# %%
# %% Cell 9: report
report = pd.DataFrame([
    dict(period=name, model=label, rmse=m[0], mae=m[1], mape=m[2], **dict(zip(("train", "val", "test"), r["sizes"])))
    for name, r in results.items()
    for label, m in (("xgb", r["model_metrics"]),) #, ("naive", r["naive_metrics"]))
])
params = pd.DataFrame({name: r["params"] for name, r in results.items()}).T

print(report.round(3).to_string(index=False))
print(params)
report.to_csv("report_metrics.csv", index=False)
params.to_csv("report_params.csv")

# %% [markdown]
# ### Checking 1 week and 1 month performance

# %%
# %% Cell 9b: 1d / 7d / 30d horizon check (no Optuna, params from results)
def horizon_scores(h, model_cls, fixed):
    cov = covariates.shift(h - 1)
    tgt = target.slice_intersect(cov)
    cov = cov.slice_intersect(tgt)
    out = {}
    for name, (s, e) in bounds.items():
        lo = pd.Timestamp(s) if s else tgt.start_time()
        hi = pd.Timestamp(e) if e else tgt.end_time()
        y, x = tgt.slice(lo, hi), cov.slice(lo, hi)
        y_fit, y_test = y.split_before(0.8)
        x_fit, _ = x.split_before(0.8)
        m = model_cls(**results[name]["params"], **fixed).fit(y_fit, past_covariates=x_fit)
        p = m.historical_forecasts(series=y, past_covariates=x, start=y_test.start_time(),
                                   forecast_horizon=h, stride=1, retrain=False,
                                   last_points_only=True, verbose=False).map(lambda v: np.round(v))
        out[name] = rmse(y_test.slice_intersect(p), p)
    return pd.Series(out)

print(pd.DataFrame({f"{h}d": horizon_scores(h, XGBModel, xgb_fixed) for h in (1, 7, 30)}).round(3))

# %% [markdown]
# ### SHAP attribute impact 

# %%
# %% Cell 10: SHAP per period
import re
from darts.explainability import ShapExplainer
import warnings
warnings.filterwarnings("ignore")


import logging
logging.getLogger("darts").setLevel(logging.ERROR)



def var_importance(r):
    res = ShapExplainer(r["model"]).explain(
        foreground_series=r["y"], foreground_past_covariates=r["x"])
    sv = res.get_explanation(horizon=1, component="AQI").to_dataframe()
    
    # mean |SHAP| on test days
    sv = sv.loc[r["y_test"].time_index].abs().mean()      
    return sv.groupby(lambda c: re.sub(r"_(target|pastcov)_lag-\d+$", "", c)).sum()

imp = pd.DataFrame({name: var_importance(r) for name, r in results.items()})

# check names match the regex
print(imp.index.tolist())      

# Grouping variables into families for better visualisation
groups = {"PM (PM10+PM25+Particles)": ["PM10", "PM25", "Particles"],
          "NOx family (NO+NO2+NOx)": ["NO", "NO2", "NOx"]}

def grouped(imp):
    g = imp.drop("AQI")
    for name, cols in groups.items():
        g.loc[name] = g.loc[cols].sum()
        g = g.drop(cols)
    return g / g.sum()


# Plot the graphs
share = grouped(imp)
ax = share.plot.bar(figsize=(12, 5), title="Share of mean |SHAP|, covariates only")
for t in ax.get_xticklabels():
    if t.get_text() in ("Radiation", "Precipitation"):
        t.set_color("red")
ax.set_title(ax.get_title() + " (red = data only in 2024)")
for c in ax.containers:
    ax.bar_label(c, fmt="%.2f", fontsize=7, padding=2)
plt.show()

ax = (imp.loc["AQI"] / imp.sum()).plot.bar(figsize=(6, 4), title="model dependency on AQI lag share")
ax.bar_label(ax.containers[0], fmt="%.2f")
plt.show()
print(f"Currently Radiation and Prepicipitation are having no data until 2024 -> they are still included but this is a limitation of the dataset. The model is not able to learn from these variables and they are not contributing to the prediction.")

# %% [markdown]
# ### Tested variable impacts of 3 models in same period

# %%
# %% Cell 12: SHAP, all 3 models on the same window
def var_importance_on(model, y_w, days=14):
    lo, hi = y_w.start_time() - pd.Timedelta(days=days), y_w.end_time()
    res = ShapExplainer(model).explain(
        foreground_series=target.slice(lo, hi),
        foreground_past_covariates=covariates.slice(lo, hi))
    sv = res.get_explanation(horizon=1, component="AQI").to_dataframe()
    sv = sv.loc[y_w.time_index].abs().mean()
    return sv.groupby(lambda c: re.sub(r"_(target|pastcov)_lag-\d+$", "", c)).sum()

# Currentlyt using the post period as the window for comparison, but can be changed to pre or during
y_w = results["pre"]["y_test"]
same = pd.DataFrame({m: var_importance_on(r["model"], y_w) for m, r in results.items()})
same_share = grouped(same)

ax = same_share.plot.bar(figsize=(12, 5),
    title=f"Same window ('{[k for k,v in results.items() if v['y_test'] is y_w][0]}' test days): share of mean |SHAP|")
for t in ax.get_xticklabels():
    if t.get_text() in ("Radiation", "Precipitation"):
        t.set_color("red")
ax.set_title(ax.get_title() + " (red = data only in 2024)")
for c in ax.containers:
    ax.bar_label(c, fmt="%.2f", fontsize=7, padding=2)
plt.show()


y_w = results["during"]["y_test"]
same = pd.DataFrame({m: var_importance_on(r["model"], y_w) for m, r in results.items()})
same_share = grouped(same)

ax = same_share.plot.bar(figsize=(12, 5),
    title=f"Same window ('{[k for k,v in results.items() if v['y_test'] is y_w][0]}' test days): share of mean |SHAP|")
for t in ax.get_xticklabels():
    if t.get_text() in ("Radiation", "Precipitation"):
        t.set_color("red")
ax.set_title(ax.get_title() + " (red = data only in 2024)")
for c in ax.containers:
    ax.bar_label(c, fmt="%.2f", fontsize=7, padding=2)
plt.show()


y_w = results["post"]["y_test"]
same = pd.DataFrame({m: var_importance_on(r["model"], y_w) for m, r in results.items()})
same_share = grouped(same)

ax = same_share.plot.bar(figsize=(12, 5),
    title=f"Same window ('{[k for k,v in results.items() if v['y_test'] is y_w][0]}' test days): share of mean |SHAP|")
for t in ax.get_xticklabels():
    if t.get_text() in ("Radiation", "Precipitation"):
        t.set_color("red")
ax.set_title(ax.get_title() + " (red = data only in 2024)")
for c in ax.containers:
    ax.bar_label(c, fmt="%.2f", fontsize=7, padding=2)
plt.show()

# %% [markdown]
# ### Beeswarm visualisation of variable impact

# %% [markdown]
# Note: Each node is the sum of SHAP value for all 7 lags, color is for lag-1 (day before)

# %%
import shap
def beeswarm(r):
    res = ShapExplainer(r["model"]).explain(
        foreground_series=r["y"], foreground_past_covariates=r["x"])
    idx = r["y_test"].time_index
    sv = res.get_explanation(horizon=1, component="AQI").to_dataframe().loc[idx]
    fv = res.get_feature_values(horizon=1, component="AQI").to_dataframe().loc[idx]
    strip = lambda c: re.sub(r"_(target|pastcov)_lag-\d+$", "", c)

    # Group the SHAP values and feature values by variable name (removing the lag suffix)
    sv = sv.T.groupby(strip).sum().T
    fv = fv.filter(regex=r"_lag-1$").rename(columns=strip)[sv.columns]
    return sv, fv

for name, r in results.items():
    sv, fv = beeswarm(r)
    shap.summary_plot(sv.values, fv, show=False)
    plt.title(name)
    plt.show()

# %% [markdown]
# ### Line graph for models' prediction vs actual data

# %% [markdown]
# #### Weekly Pred vs Actual

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharey=True)
colors = dict(pre="tab:blue", during="tab:orange", post="tab:green")

for ax, (name, r) in zip(axes, results.items()):
    actual_w = r["y_test"].to_series().resample("W").mean()
    pred_w = r["test_pred"].to_series().resample("W").mean()

    ax.plot(actual_w.index, actual_w.values, color="black", lw=2, marker="o", markersize=4, label="actual", zorder=2)
    ax.plot(pred_w.index, pred_w.values, color=colors[name], lw=1.2, marker="o", markersize=3,
        linestyle="--", label="predicted", zorder=3)
    ax.set_title(name)
    ax.tick_params(axis="x", rotation=45)
    ax.legend()

    

axes[0].set_ylabel("AQI (weekly mean)")
plt.tight_layout()
plt.show()

# %% [markdown]
# #### Daily Pred vs Actual

# %%
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharey=True)
colors = dict(pre="tab:blue", during="tab:orange", post="tab:green")

for ax, (name, r) in zip(axes, results.items()):
    actual, pred = r["y_test"].to_series(), r["test_pred"].to_series()
    ax.plot(actual.index, actual.values, color="black", lw=1.5, marker="o", markersize=3, label="actual")
    ax.plot(pred.index, pred.values, color=colors[name], lw=1, marker="o", markersize=2,
            linestyle="--", label="predicted")
    ax.set_title(f"{name}-daily period: {r['sizes'][2]} test days, RMSE={r['model_metrics'][0]:.2f}, MAE={r['model_metrics'][1]:.2f}")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()

axes[0].set_ylabel("AQI (daily)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### What pollutant dominates each periods?

# %%
idx = dominant.index.tz_localize(None)
per = np.where(idx <= "2020-02-29", "pre", np.where(idx <= "2021-04-30", "during", "post"))
(dominant.groupby(per).value_counts(normalize=True).unstack()
    .loc[["pre", "during", "post"]].plot.bar(figsize=(8, 4), title="Dominant pollutant share of days"))

# %% [markdown]
# ### Correlation between each covariate in 3 periods

# %%
fig, axes = plt.subplots(3, 1, figsize=(80, 40))

for ax, (n, (s, e)) in zip(axes, bounds.items()):
    c = covariates.slice(
        pd.Timestamp(s) if s else covariates.start_time(),
        pd.Timestamp(e) if e else covariates.end_time()
    ).to_dataframe().corr()
    
    im = ax.imshow(c, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title(n, fontsize=14, pad=10)
    
    ax.set_xticks(range(len(c)))
    ax.set_xticklabels(c.columns, rotation=90)
    ax.set_yticks(range(len(c)))
    ax.set_yticklabels(c.columns)
    
    # Add numerical values to each heatmap cell
    for i in range(len(c)):
        for j in range(len(c)):
            val = c.iloc[i, j]
            # Change text color to white for dark backgrounds for contrast
            text_color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)

# Increase vertical spacing (padding) between subplots
plt.subplots_adjust(hspace=0.2)

plt.show()

# %%



