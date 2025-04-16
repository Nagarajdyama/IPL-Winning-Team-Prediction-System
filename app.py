import streamlit as st
import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import shap
import lime
import lime.lime_tabular
import requests
import time
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Page Configuration & Title
# -----------------------------
st.set_page_config(page_title="💎 Ultimate IPL Predictor – Multi-file Enhanced", layout="wide")
st.title("🏏 Ultimate IPL Winning Team Predictor – Multi-file Enhanced Edition")


# -----------------------------
# Load and Process Deliveries Data
# -----------------------------
@st.cache_data(show_spinner=False)
def load_deliveries():
    """
    Loads deliveries.csv and computes team-level batting and bowling statistics.
    Expected columns:
      - batting_team, batsman_runs, ball  (for batting metrics)
      - bowling_team, total_runs, ball, dismissal_kind (for bowling metrics)
    """
    try:
        df = pd.read_csv("deliveries.csv")

        # --- Aggregate Team-Level Batting Metrics ---
        if {"batting_team", "batsman_runs", "ball"}.issubset(df.columns):
            team_batting = df.groupby("batting_team").agg({
                "batsman_runs": "sum",
                "ball": "count"
            }).rename(columns={"batsman_runs": "team_total_runs", "ball": "team_balls_faced"})
            team_batting["team_strike_rate"] = (team_batting["team_total_runs"] / team_batting[
                "team_balls_faced"]) * 100
        else:
            team_batting = None

        # --- Aggregate Team-Level Bowling Metrics ---
        if {"bowling_team", "total_runs", "ball", "dismissal_kind"}.issubset(df.columns):
            team_bowling = df.groupby("bowling_team").agg({
                "total_runs": "sum",
                "ball": "count",
                "dismissal_kind": "count"
            }).rename(columns={
                "total_runs": "team_runs_conceded",
                "ball": "team_balls_bowled",
                "dismissal_kind": "team_wickets"
            })
            team_bowling["team_economy_rate"] = (team_bowling["team_runs_conceded"] / team_bowling[
                "team_balls_bowled"]) * 6
        else:
            team_bowling = None

        return {"batting": team_batting, "bowling": team_bowling, "raw": df}
    except Exception as e:
        st.error(f"Error loading deliveries.csv: {e}")
        return None


deliveries_data = load_deliveries()

# -----------------------------
# Sidebar: User Settings & Advanced Options
# -----------------------------
st.sidebar.header("🗂 Upload Additional IPL Datasets")
data_files = st.sidebar.file_uploader("Upload multiple IPL datasets (CSV/Excel)",
                                      type=["csv", "xlsx"], accept_multiple_files=True)

st.sidebar.header("🛠 Model Selection & Optimization")
model_choice = st.sidebar.selectbox("Select Model",
                                    ["Random Forest", "XGBoost", "LightGBM", "CatBoost", "Neural Network"])
optimize_hyperparams = st.sidebar.checkbox("Optimize Hyperparameters with Optuna?", value=True)


# -----------------------------
# Utility Function: Load Additional Files
# -----------------------------
@st.cache_data(show_spinner=False)
def load_multiple_files(files):
    """Loads and merges multiple CSV/Excel files."""
    dfs = []
    for file in files:
        try:
            if file.name.endswith('.csv'):
                dfs.append(pd.read_csv(file))
            elif file.name.endswith('.xlsx'):
                dfs.append(pd.read_excel(file, engine='openpyxl'))
        except Exception as e:
            st.error(f"Error loading {file.name}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else None


def preprocess_data(df):
    """
    Processes the main dataset by dropping duplicates and rows missing critical columns.
    Critical columns: "winner", "team1", and "team2".
    Stores original team names and creates numeric encodings.
    """
    df = df.drop_duplicates().reset_index(drop=True)
    critical_cols = ["winner", "team1", "team2"]
    if set(critical_cols).issubset(df.columns):
        df = df.dropna(subset=critical_cols)
    else:
        st.error("Dataset missing one or more critical columns: 'winner', 'team1', 'team2'.")
        return df, {}

    # Store original team names for merging later
    df["team1_orig"] = df["team1"]
    df["team2_orig"] = df["team2"]

    # Create numeric encodings for team names for model training
    le_team1 = LabelEncoder()
    df["team1_enc"] = le_team1.fit_transform(df["team1"])
    le_team2 = LabelEncoder()
    df["team2_enc"] = le_team2.fit_transform(df["team2"])

    # Encode additional columns: toss_winner, venue, and winner
    encoder_map = {"team1": le_team1, "team2": le_team2}
    for col in ["toss_winner", "venue", "winner"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoder_map[col] = le

    # Create aggregate features
    df["team1_wins"] = df.groupby("team1_orig")["winner"].transform(lambda x: (x == 0).sum())
    df["team2_wins"] = df.groupby("team2_orig")["winner"].transform(lambda x: (x == 1).sum())
    df["match_diff"] = df["team1_wins"] - df["team2_wins"]
    if {"venue", "winner"}.issubset(df.columns):
        df["venue_strength"] = df["venue"].map(df.groupby("venue")["winner"].mean())
    else:
        df["venue_strength"] = 0

    st.write("Data shape after preprocessing:", df.shape)
    return df, encoder_map


def train_model(df, model_choice):
    """
    Trains a prediction model using features from the main dataset and aggregated team-level stats
    from the deliveries data. Merges aggregated batting and bowling metrics using original team names.
    """
    # Use encoded team names for modeling
    available_columns = ["team1_enc", "team2_enc", "toss_winner", "team1_wins", "team2_wins", "match_diff",
                         "venue_strength"]
    optional_columns = ["humidity", "temperature"]
    selected_columns = available_columns + [col for col in optional_columns if col in df.columns]

    # Check for original team names for merging aggregated features
    if "team1_orig" not in df.columns or "team2_orig" not in df.columns:
        st.error("Original team name columns not found for merging aggregated features.")
        return None

    # --- Merge Team-Level Batting Metrics ---
    if deliveries_data is not None and deliveries_data.get("batting") is not None:
        team_batting = deliveries_data["batting"].reset_index()  # 'batting_team' becomes a column
        df = df.merge(team_batting[["batting_team", "team_strike_rate"]], how="left",
                      left_on="team1_orig", right_on="batting_team") \
            .rename(columns={"team_strike_rate": "team1_strike_rate"}) \
            .drop(columns=["batting_team"], errors="ignore")
        df = df.merge(team_batting[["batting_team", "team_strike_rate"]], how="left",
                      left_on="team2_orig", right_on="batting_team") \
            .rename(columns={"team_strike_rate": "team2_strike_rate"}) \
            .drop(columns=["batting_team"], errors="ignore")
        selected_columns += ["team1_strike_rate", "team2_strike_rate"]

    # --- Merge Team-Level Bowling Metrics ---
    if deliveries_data is not None and deliveries_data.get("bowling") is not None:
        team_bowling = deliveries_data["bowling"].reset_index()  # 'bowling_team' here
        df = df.merge(team_bowling[["bowling_team", "team_economy_rate", "team_wickets"]], how="left",
                      left_on="team1_orig", right_on="bowling_team") \
            .rename(columns={"team_economy_rate": "team1_economy_rate",
                             "team_wickets": "team1_wickets"}) \
            .drop(columns=["bowling_team"], errors="ignore")
        df = df.merge(team_bowling[["bowling_team", "team_economy_rate", "team_wickets"]], how="left",
                      left_on="team2_orig", right_on="bowling_team") \
            .rename(columns={"team_economy_rate": "team2_economy_rate",
                             "team_wickets": "team2_wickets"}) \
            .drop(columns=["bowling_team"], errors="ignore")
        selected_columns += ["team1_economy_rate", "team1_wickets", "team2_economy_rate", "team2_wickets"]

    if len(df) < 2:
        st.error("Not enough data for training. Check your data preprocessing!")
        return None

    X = df[selected_columns]
    y = df["winner"] if "winner" in df.columns else None
    if y is None or X.empty:
        st.error("The main dataset is missing required columns for prediction.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        "LightGBM": LGBMClassifier(),
        "CatBoost": CatBoostClassifier(verbose=0),
        "Neural Network": MLPClassifier(max_iter=500)
    }
    model = models[model_choice]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"{model_choice} Accuracy: {acc:.2f}")
    return model


# -----------------------------
# Main Execution Flow
# -----------------------------
df = None  # Initialize to avoid NameError later.
if data_files:
    df = load_multiple_files(data_files)
    if df is not None:
        st.write("Columns in main dataset:", df.columns)
        df, encoder_map = preprocess_data(df)
        model = train_model(df, model_choice)
    else:
        st.warning("No valid datasets were loaded.")
else:
    st.info("Upload IPL datasets (CSV or Excel) to get started!")

# -----------------------------
# Interactive Match Prediction Section with Advanced Animation and Balloons
# -----------------------------
if df is not None and "team1_orig" in df.columns:
    st.markdown("## Predict a Match Winner")
    teams = df["team1_orig"].unique()
    selected_team1 = st.selectbox("Select Team 1", teams)
    teams_for_team2 = [team for team in teams if team != selected_team1]
    selected_team2 = st.selectbox("Select Team 2", teams_for_team2)
    toss_winner = st.radio("Who won the toss?", (selected_team1, selected_team2))

    if st.button("Predict Winner"):
        with st.spinner("Analyzing match, please wait..."):
            time.sleep(3)  # Simulate analysis delay

        # Retrieve encoded team values if available.
        if encoder_map and "team1" in encoder_map and "team2" in encoder_map:
            team1_enc = encoder_map["team1"].transform([selected_team1])[0]
            team2_enc = encoder_map["team2"].transform([selected_team2])[0]
            toss_enc = encoder_map["toss_winner"].transform([toss_winner])[0] if "toss_winner" in encoder_map else 0
        else:
            team1_enc, team2_enc, toss_enc = 0, 0, 0

        # For demonstration, we assign default values to aggregated features.
        team1_wins = 10
        team2_wins = 10
        match_diff = team1_wins - team2_wins
        venue_strength = 0.5
        team1_strike_rate = 130
        team2_strike_rate = 125
        team1_economy_rate = 7.0
        team1_wickets = 15
        team2_economy_rate = 7.5
        team2_wickets = 14

        # Create a DataFrame for the sample match using the same feature order as training.
        features = {
            "team1_enc": [team1_enc],
            "team2_enc": [team2_enc],
            "toss_winner": [toss_enc],
            "team1_wins": [team1_wins],
            "team2_wins": [team2_wins],
            "match_diff": [match_diff],
            "venue_strength": [venue_strength],
            "team1_strike_rate": [team1_strike_rate],
            "team2_strike_rate": [team2_strike_rate],
            "team1_economy_rate": [team1_economy_rate],
            "team1_wickets": [team1_wickets],
            "team2_economy_rate": [team2_economy_rate],
            "team2_wickets": [team2_wickets]
        }
        X_new = pd.DataFrame(features)
        predicted = model.predict(X_new)
        if encoder_map and "winner" in encoder_map:
            predicted_winner = encoder_map["winner"].inverse_transform(predicted)[0]
        else:
            predicted_winner = predicted[0]

        # Trigger default balloon animation.
        st.balloons()

        # Advanced animated HTML: Initially the predicted team name appears huge (scale(2))
        # then smoothly shrinks to the final size (scale(1)).
        animated_html = f"""
        <style>
        @keyframes shrinkAnimation {{
            0% {{
                transform: scale(2);
                opacity: 1;
            }}
            100% {{
                transform: scale(1);
                opacity: 1;
            }}
        }}
        @keyframes shimmer {{
            0% {{
                background-position: -500px 0;
            }}
            100% {{
                background-position: 500px 0;
            }}
        }}
        .winner-container {{
            text-align: center;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            margin-top: 20px;
        }}
        .winner-label {{
            font-size: 18px;
            color: #2E86C1;
        }}
        .winner-name {{
            display: inline-block;
            font-size: 40px;
            font-weight: bold;
            color: #D35400;
            padding: 10px 20px;
            border-radius: 10px;
            animation: shrinkAnimation 5s ease-out forwards, shimmer 3s linear infinite;
            background: linear-gradient(90deg, #D35400, #F39C12, #D35400);
            background-size: 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        </style>
        <div class="winner-container">
            <div class="winner-label">Predicted Winner:</div>
            <div class="winner-name">{predicted_winner}</div>
        </div>
        """
        st.markdown(animated_html, unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with ❤️ by Nagaraj Dyama | Multi-file AI-Powered Predictor")
