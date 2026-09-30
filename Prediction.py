#Logistic Regression Model

import pyodbc
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import nflreadpy as nfl
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error
# =========================================================
# 1. KEEP ONLY LAST 5 REGULAR SEASONS
# =========================================================
seasons = list(range(2021, 2026))
pbp = nfl.load_pbp(seasons=seasons).to_pandas()
pbp = pbp.copy()

pbp = pbp[pbp["season_type"] == "REG"].copy()

# =========================================================
# 2. DEFINE CLUTCH (UPDATED VERSION)
# =========================================================
pbp["is_clutch"] = (
    (
        (pbp["qtr"] == 4) &
        (pbp["quarter_seconds_remaining"] <= 480) &
        (pbp["score_differential"] <=8) &
        (pbp["score_differential"] >= -8)
    )
    |
    (
        (pbp["qtr"] == 5) &
        (pbp["score_differential"] <= 8) &
        (pbp["score_differential"] >= -8)
    )
)

pbp["is_baseline"] = ~pbp["is_clutch"]

pbp["turnover"] = (
    (pbp["interception"] == 1) |
    (pbp["fumble_lost"] == 1)
).astype(int)
# =========================================================
# 3. BUILD TEAM-SEASON METRICS
# =========================================================
playoff_dict = {
    2021: ['KC', 'BUF', 'TEN', 'CIN', 'NE', 'LV', 'PIT',
           'GB', 'TB', 'DAL', 'LA', 'ARI', 'SF', 'PHI'], 
    2022: ['KC', 'BUF', 'CIN', 'JAX', 'LAC', 'BAL', 'MIA',
           'PHI', 'SF', 'MIN', 'TB', 'DAL', 'NYG', 'SEA'],
    2023: ['BAL', 'BUF', 'HOU', 'KC', 'CLE', 'MIA', 'PIT',
           'DET', 'SF', 'LA', 'PHI', 'DAL', 'GB', 'TB'],
    2024: ['BAL', 'BUF', 'HOU', 'KC', 'DEN', 'PIT', 'LAC',
           'DET', 'TB', 'LA', 'PHI', 'MIN', 'GB', 'WAS'],
    2025: ['SEA', 'BUF', 'HOU', 'DEN', 'NE', 'LA', 'LAC',
           'PIT', 'JAX', 'CHI', 'PHI', 'SF', 'GB', 'CAR']
}
rows = []

for season, teams in playoff_dict.items():
    for team in teams:
        rows.append({"season": season, "posteam": team, "made_playoffs": 1})

playoff_df = pd.DataFrame(rows)
overall = (
    pbp.groupby(["season", "posteam"], as_index=False)
    .agg(
        epa_per_play=("epa", "mean"),
        wpa_per_play=("wpa", "mean"),
        play_count=("epa", "count")
    )
)

clutch = (
    pbp[pbp["is_clutch"]]
    .groupby(["season", "posteam"], as_index=False)
    .agg(
        clutch_epa_per_play=("epa", "mean"),
        Clutch_wpa_per_play=("wpa", "mean"),
        clutch_plays=("epa", "count")
    )
)

baseline = (
    pbp[pbp["is_baseline"]]
    .groupby(["season", "posteam"], as_index=False)
    .agg(
        baseline_epa_per_play=("epa", "mean"),
        baseline_wpa_per_play=("wpa", "mean")
    )
)

df = overall.merge(clutch, on=["season", "posteam"], how="left")
df = df.merge(baseline, on=["season", "posteam"], how="left")

df["clutch_score"] = df["clutch_epa_per_play"] - df["baseline_epa_per_play"]
df["clutch_wpa_score"] = df["Clutch_wpa_per_play"] - df["baseline_wpa_per_play"]

# =========================================================
# 4. CREATE TEAM WINS FROM PBP
# =========================================================

games = (
    pbp.groupby("game_id", as_index=False)
    .agg(
        season=("season", "first"),
        home_team=("home_team", "first"),
        away_team=("away_team", "first"),
        total_home_score=("total_home_score", "max"),
        total_away_score=("total_away_score", "max")
    )
)
# =========================================================
# POINT DIFFERENTIAL PER GAME
# =========================================================

# Calculate point differential at game level
games["point_diff"] = games["total_home_score"] - games["total_away_score"]

# Home team perspective
home_pd = games[["season", "home_team", "point_diff"]].rename(
    columns={"home_team": "posteam"}
)

# Away team perspective (flip sign)
away_pd = games[["season", "away_team", "point_diff"]].copy()
away_pd["point_diff"] = -away_pd["point_diff"]
away_pd = away_pd.rename(columns={"away_team": "posteam"})

# Combine both
point_diff_df = pd.concat([home_pd, away_pd], ignore_index=True)

# Aggregate to team-season
point_diff_df = (
    point_diff_df.groupby(["season", "posteam"], as_index=False)
    .agg(point_diff=("point_diff", "sum"))
)

# Convert to per game
point_diff_df["point_diff_per_game"] = point_diff_df["point_diff"] / 17

games["home_win"] = (games["total_home_score"] > games["total_away_score"]).astype(int)
games["away_win"] = (games["total_away_score"] > games["total_home_score"]).astype(int)

home_wins = games[["season", "home_team", "home_win"]].rename(
    columns={"home_team": "posteam", "home_win": "wins"}
)

away_wins = games[["season", "away_team", "away_win"]].rename(
    columns={"away_team": "posteam", "away_win": "wins"}
)

wins = pd.concat([home_wins, away_wins])
wins = wins.groupby(["season", "posteam"], as_index=False)["wins"].sum()

df = df.merge(wins, on=["season", "posteam"], how="left")
df = df.merge(
    point_diff_df[["season", "posteam", "point_diff_per_game"]],
    on=["season", "posteam"],
    how="left"
)

giveaways = (
    pbp[pbp["turnover"] == 1]
    .groupby(["season", "posteam"], as_index=False)
    .agg(giveaways=("turnover", "sum"))
)

takeaways = (
    pbp[pbp["turnover"] == 1]
    .groupby(["season", "defteam"], as_index=False)
    .agg(takeaways=("turnover", "sum"))
    .rename(columns={"defteam": "posteam"})
)
turnover_df = giveaways.merge(
    takeaways,
    on=["season", "posteam"],
    how="outer"
).fillna(0)

turnover_df["turnover_diff"] = (
    turnover_df["takeaways"] - turnover_df["giveaways"]
)

turnover_df["turnover_diff_per_game"] = turnover_df["turnover_diff"] / 17

# Merge turnover stats
df = df.merge(
    turnover_df[["season", "posteam", "giveaways", "takeaways", "turnover_diff", "turnover_diff_per_game"]],
    on=["season", "posteam"],
    how="left"
)
df = df.merge(
    playoff_df,
    on=["season", "posteam"],
    how="left"
)

df["made_playoffs"] = df["made_playoffs"].fillna(0)
# =========================================================
# 5. TRAIN / TEST SPLIT
# =========================================================

train_df = df[df["season"].between(2021, 2024)].copy()
test_df = df[df["season"] == 2025].copy()

# Remove missing values
train_df = train_df.dropna(subset=["epa_per_play", "clutch_score", "wins"])
test_df = test_df.dropna(subset=["epa_per_play", "clutch_score", "wins"])

# Optional: remove small samples
train_df = train_df[train_df["clutch_plays"] >= 20]
test_df = test_df[test_df["clutch_plays"] >= 20]

# =========================================================
# 6. TRAIN MODELS
# =========================================================

# Model 1: Baseline

model3 = smf.ols("wins ~  Clutch_wpa_per_play+ point_diff_per_game", data=train_df).fit()
# Model 2: Add clutch
modelLogisitc=smf.logit( formula="""
    made_playoffs ~ Clutch_wpa_per_play+point_diff_per_game """,data=train_df).fit()



print(modelLogisitc.summary())
test_df["pred_prob"] = modelLogisitc.predict(test_df)
test_df["pred_class"] = (test_df["pred_prob"] > 0.5).astype(int)
accuracy = (test_df["pred_class"] == test_df["made_playoffs"]).mean()
print("Test Accuracy:", accuracy)
# Compare counts


cm = pd.crosstab(
    test_df["made_playoffs"],
    test_df["pred_class"],
    rownames=["Actual"],
    colnames=["Predicted"]
)

print(cm)
print(model3.summary())



X = df[[
    "baseline_wpa_per_play",
    "Clutch_wpa_per_play",
    "point_diff_per_game"
]].copy()
X = sm.add_constant(X)
vif_data = pd.DataFrame()
vif_data["variable"] = X.columns
vif_data["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

print(vif_data)
# =========================================================
# 7. PREDICT ON 2026
# =========================================================

test_df["predicted_wins"] = model3.predict(test_df)
rmse = np.sqrt(mean_squared_error(
    test_df["wins"],
    test_df["predicted_wins"]
))

print("RMSE:", rmse)
rmse_train = model3.mse_resid ** 0.5
print("Training RMSE:", rmse_train)

print(df[["Clutch_wpa_per_play", "point_diff_per_game"]].corr())



import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import numpy as np

# --- TRAIN METRICS ---
train_r2 = model3.rsquared
train_adj_r2 = model3.rsquared_adj
train_rmse = np.sqrt(model3.mse_resid)

# --- TEST METRICS ---
y_true = test_df["wins"]
y_pred = test_df["predicted_wins"]

# RMSE
test_rmse = np.sqrt(np.mean((y_true - y_pred)**2))

# R²
ss_res = np.sum((y_true - y_pred)**2)
ss_tot = np.sum((y_true - np.mean(y_true))**2)
test_r2 = 1 - (ss_res / ss_tot)

# Adjusted R² for test
n = len(y_true)
k = model3.df_model  # number of predictors

test_adj_r2 = 1 - (1 - test_r2) * (n - 1) / (n - k - 1)

# --- CREATE TABLE ---
comparison_table = pd.DataFrame({
    "Dataset": ["Training", "Test"],
    "R-squared": [train_r2, test_r2],
    "Adj. R-squared": [train_adj_r2, test_adj_r2],
    "RMSE": [train_rmse, test_rmse]
})

# Round for clean output
comparison_table = comparison_table.round(3)

print(comparison_table)
