# app.py
from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import streamlit as st

import matplotlib.pyplot as plt

from passlib.context import CryptContext

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    silhouette_score,
)

# ============================================================
# CONFIG
# ============================================================
@dataclass(frozen=True)
class Config:
    default_data_path: str = "SmartCoach_merged.xlsx"  # set env ASSISTANT_COACH_DATA_PATH to override
    models_dir: str = "models"
    data_dir: str = "data"
    auth_db_path: str = "auth/users.db"
    random_state: int = 42
    test_size: float = 0.2

    reg_model_file: str = "goals_regressor.joblib"
    clf_model_file: str = "fatigue_classifier.joblib"
    km_model_file: str = "player_kmeans.joblib"
    meta_file: str = "metrics_meta.joblib"


CFG = Config()

# Password hashing (bcrypt)
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


# ============================================================
# AUTH (SQLite + bcrypt)
# ============================================================
def ensure_auth_db() -> None:
    db_path = Path(CFG.auth_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT,
                password_hash TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.commit()


def is_valid_username(username: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_]{3,20}", username.strip()))


def is_strong_password(password: str) -> Tuple[bool, str]:
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if not re.search(r"[A-Z]", password):
        return False, "Password must include an uppercase letter."
    if not re.search(r"[a-z]", password):
        return False, "Password must include a lowercase letter."
    if not re.search(r"[0-9]", password):
        return False, "Password must include a number."
    return True, ""


def create_user(username: str, email: str, password: str) -> Tuple[bool, str]:
    ensure_auth_db()

    username = username.strip()
    email = email.strip()

    if not is_valid_username(username):
        return False, "Username must be 3–20 characters (letters, numbers, underscore)."

    ok, msg = is_strong_password(password)
    if not ok:
        return False, msg

    password_hash = pwd_context.hash(password)

    try:
        with sqlite3.connect(CFG.auth_db_path) as conn:
            conn.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email if email else None, password_hash),
            )
            conn.commit()
        return True, "Account created successfully. You can now sign in."
    except sqlite3.IntegrityError:
        return False, "That username already exists. Please choose another."
    except Exception as e:
        return False, f"Failed to create account: {e}"


def verify_user(username: str, password: str) -> Tuple[bool, str]:
    ensure_auth_db()
    username = username.strip()

    with sqlite3.connect(CFG.auth_db_path) as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()

    if row is None:
        return False, "Invalid username or password."

    stored_hash = row[0]
    if not pwd_context.verify(password, stored_hash):
        return False, "Invalid username or password."

    return True, "Signed in successfully."


def any_users_exist() -> bool:
    ensure_auth_db()
    with sqlite3.connect(CFG.auth_db_path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM users").fetchone()
    return bool(row and row[0] > 0)


def require_auth() -> Optional[str]:
    """
    Returns the authenticated username or None.
    Shows auth UI if not logged in.
    """
    if "auth" not in st.session_state:
        st.session_state.auth = {"logged_in": False, "username": None}

    if st.session_state.auth.get("logged_in"):
        return st.session_state.auth.get("username")

    st.title("⚽ Assistant Coach")
    st.info("Please sign in to use the app.")

    tab_signin, tab_signup = st.tabs(["Sign in", "Sign up"])

    with tab_signin:
        u = st.text_input("Username", key="signin_user")
        p = st.text_input("Password", type="password", key="signin_pass")
        if st.button("Sign in", type="primary", use_container_width=True):
            ok, msg = verify_user(u, p)
            if ok:
                st.session_state.auth = {"logged_in": True, "username": u.strip()}
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    with tab_signup:
        if any_users_exist():
            st.caption("Create a new account (coach / staff).")
        else:
            st.warning(
                "No users exist yet. Create the first account now (this will become the first login)."
            )

        u2 = st.text_input("Username (3–20 chars, letters/numbers/_)", key="signup_user")
        e2 = st.text_input("Email (optional)", key="signup_email")
        p2 = st.text_input("Password", type="password", key="signup_pass")
        p3 = st.text_input("Confirm password", type="password", key="signup_pass2")

        if st.button("Create account", use_container_width=True):
            if p2 != p3:
                st.error("Passwords do not match.")
            else:
                ok, msg = create_user(u2, e2, p2)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    return None


def logout_button():
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.auth = {"logged_in": False, "username": None}
        st.rerun()


# ============================================================
# DATA LOADING & VALIDATION
# ============================================================
REQUIRED_COLUMNS = [
    "Player",
    "Team",
    "Opponent",
    "Match_Date",
    "Goals",
    "Assists",
    "Pass_Accuracy_%",
    "Minutes_Played",
    "Fouls",
    "Shots_on_Target",
    "Fatigue_Level",
]


def get_data_path() -> Path:
    p = os.environ.get("ASSISTANT_COACH_DATA_PATH", CFG.default_data_path)
    return Path(p)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}. Upload it in the sidebar or set ASSISTANT_COACH_DATA_PATH."
        )

    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported file type. Use .xlsx or .csv")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Dataset is missing required columns:\n" + "\n".join(f"- {m}" for m in missing))

    df = df.copy()
    df["Match_Date"] = pd.to_datetime(df["Match_Date"], errors="coerce")

    # Coerce numeric columns
    numeric_cols = ["Goals", "Assists", "Pass_Accuracy_%", "Minutes_Played", "Fouls", "Shots_on_Target"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Standardize Fatigue_Level strings
    df["Fatigue_Level"] = df["Fatigue_Level"].astype(str).str.strip().str.title()
    df.loc[~df["Fatigue_Level"].isin(["Low", "Medium", "High"]), "Fatigue_Level"] = np.nan

    return df


# ============================================================
# FEATURE SETS (AS PER YOUR REPORT)
# ============================================================
REG_FEATURES = ["Minutes_Played", "Shots_on_Target", "Pass_Accuracy_%", "Fatigue_Level", "Assists"]
REG_TARGET = "Goals"

CLF_FEATURES = ["Goals", "Assists", "Pass_Accuracy_%", "Minutes_Played", "Fouls", "Shots_on_Target"]
CLF_TARGET = "Fatigue_Level"

KM_FEATURES = ["Goals", "Assists", "Pass_Accuracy_%", "Shots_on_Target"]


# ============================================================
# MODEL BUILDING
# ============================================================
def make_regression_pipeline() -> Pipeline:
    numeric = ["Minutes_Played", "Shots_on_Target", "Pass_Accuracy_%", "Assists"]
    categorical = ["Fatigue_Level"]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=CFG.random_state,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    return Pipeline([("preprocess", pre), ("model", model)])


def make_classification_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), CLF_FEATURES),
        ],
        remainder="drop",
    )

    model = DecisionTreeClassifier(
        random_state=CFG.random_state,
        max_depth=6,
        min_samples_leaf=8,
    )

    return Pipeline([("preprocess", pre), ("model", model)])


def make_kmeans_pipeline(n_clusters: int = 3) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), KM_FEATURES),
        ],
        remainder="drop",
    )

    km = KMeans(n_clusters=n_clusters, random_state=CFG.random_state, n_init=20)
    return Pipeline([("preprocess", pre), ("model", km)])


# ============================================================
# TRAIN & PERSIST
# ============================================================
def ensure_models_dir() -> Path:
    d = Path(CFG.models_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_paths() -> dict[str, Path]:
    d = ensure_models_dir()
    return {
        "reg": d / CFG.reg_model_file,
        "clf": d / CFG.clf_model_file,
        "km": d / CFG.km_model_file,
        "meta": d / CFG.meta_file,
    }


def train_all(df: pd.DataFrame) -> dict:
    # Regression
    X_reg = df[REG_FEATURES].copy()
    y_reg = df[REG_TARGET].copy()

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg, y_reg, test_size=CFG.test_size, random_state=CFG.random_state
    )
    reg_pipe = make_regression_pipeline()
    reg_pipe.fit(Xr_train, yr_train)

    pred_reg = reg_pipe.predict(Xr_test)
    rmse = float(np.sqrt(mean_squared_error(yr_test, pred_reg)))
    mae = float(mean_absolute_error(yr_test, pred_reg))
    r2 = float(r2_score(yr_test, pred_reg))

    # Classification
    X_clf = df[CLF_FEATURES].copy()
    y_clf = df[CLF_TARGET].copy()

    mask = y_clf.notna()
    X_clf = X_clf.loc[mask]
    y_clf = y_clf.loc[mask]

    if y_clf.nunique() < 2:
        raise ValueError("Not enough class variety in Fatigue_Level to train classifier (need at least 2 classes).")

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_clf, y_clf,
        test_size=CFG.test_size,
        random_state=CFG.random_state,
        stratify=y_clf,
    )
    clf_pipe = make_classification_pipeline()
    clf_pipe.fit(Xc_train, yc_train)

    pred_clf = clf_pipe.predict(Xc_test)
    acc = float(accuracy_score(yc_test, pred_clf))
    labels = sorted(list(y_clf.unique()))
    cm = confusion_matrix(yc_test, pred_clf, labels=labels).tolist()

    # KMeans
    km_pipe = make_kmeans_pipeline(n_clusters=3)
    km_pipe.fit(df[KM_FEATURES].copy())
    clusters = km_pipe.predict(df[KM_FEATURES].copy())

    X_scaled = km_pipe.named_steps["preprocess"].transform(df[KM_FEATURES].copy())
    sil = float(silhouette_score(X_scaled, clusters)) if len(set(clusters)) > 1 else float("nan")

    meta = {
        "regression": {"rmse": rmse, "mae": mae, "r2": r2},
        "classification": {"accuracy": acc, "confusion_matrix": cm, "labels": labels},
        "clustering": {"silhouette": sil, "n_clusters": 3},
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
    }

    paths = model_paths()
    joblib.dump(reg_pipe, paths["reg"])
    joblib.dump(clf_pipe, paths["clf"])
    joblib.dump(km_pipe, paths["km"])
    joblib.dump(meta, paths["meta"])

    return meta


def load_models() -> tuple[Pipeline, Pipeline, Pipeline, dict]:
    paths = model_paths()
    if not (paths["reg"].exists() and paths["clf"].exists() and paths["km"].exists() and paths["meta"].exists()):
        raise FileNotFoundError("Models not trained yet.")
    return (
        joblib.load(paths["reg"]),
        joblib.load(paths["clf"]),
        joblib.load(paths["km"]),
        joblib.load(paths["meta"]),
    )


# ============================================================
# STREAMLIT APP
# ============================================================
st.set_page_config(page_title="Assistant Coach", page_icon="⚽", layout="wide")

# Auth gate
user = require_auth()
if user is None:
    st.stop()

with st.sidebar:
    st.success(f"Signed in as **{user}**")
    logout_button()

st.title("⚽ Assistant Coach")
st.caption("Goals prediction (regression), Readiness prediction (fatigue classification), and player clustering — with analytics plots.")

# Sidebar dataset & training controls
with st.sidebar:
    st.header("Dataset & Models")
    st.write("Upload a dataset (xlsx/csv) or use `ASSISTANT_COACH_DATA_PATH`.")

    uploaded = st.file_uploader("Upload dataset", type=["xlsx", "xls", "csv"])

    data_path = get_data_path()
    if uploaded is not None:
        save_dir = Path(CFG.data_dir)
        save_dir.mkdir(exist_ok=True)
        out_path = save_dir / uploaded.name
        out_path.write_bytes(uploaded.getvalue())
        data_path = out_path
        st.success(f"Using uploaded file: {out_path}")

    retrain = st.button("🔁 Retrain models", use_container_width=True)
    st.caption("Retrain after updating the dataset.")

# Load data
try:
    df = load_dataset(data_path)
except Exception as e:
    st.error(str(e))
    st.stop()

# Train if needed
paths = model_paths()
need_train = retrain or not (paths["reg"].exists() and paths["clf"].exists() and paths["km"].exists() and paths["meta"].exists())
if need_train:
    with st.spinner("Training models..."):
        try:
            meta = train_all(df)
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()
    st.success("Models trained and saved.")

# Load models
try:
    reg_model, clf_model, km_model, meta = load_models()
except Exception as e:
    st.error(str(e))
    st.stop()

tabs = st.tabs([
    "🏠 Home",
    "🎯 Predict Goals",
    "🧠 Predict Readiness",
    "🧩 Player Clusters",
    "📈 Visual Analytics",
    "📊 Data Explorer",
    "🛠️ Model Metrics",
])

# -----------------------------
# Home
# -----------------------------
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", meta.get("rows", len(df)))
    c2.metric("Regression RMSE", f"{meta['regression']['rmse']:.2f}")
    c3.metric("Classification Accuracy", f"{meta['classification']['accuracy']*100:.1f}%")

    st.subheader("Quick look")
    st.dataframe(df.head(25), use_container_width=True)


# -----------------------------
# Predict Goals (Regression)
# -----------------------------
with tabs[1]:
    st.subheader("🎯 Predict Goals (Regression)")
    st.caption("Inputs: Minutes_Played, Shots_on_Target, Pass_Accuracy_%, Fatigue_Level, Assists")

    colA, colB = st.columns([1, 1])

    with colA:
        minutes = st.number_input("Minutes Played", min_value=0, max_value=130, value=75)
        shots = st.number_input("Shots on Target", min_value=0, max_value=20, value=2)
        pass_acc = st.number_input("Pass Accuracy (%)", min_value=0.0, max_value=100.0, value=82.0, step=0.1)
        assists = st.number_input("Assists", min_value=0, max_value=10, value=0)
        fatigue = st.selectbox("Fatigue Level", options=["Low", "Medium", "High"], index=1)
        predict_goals = st.button("Predict Goals", type="primary", use_container_width=True)

    with colB:
        if predict_goals:
            X = pd.DataFrame([{
                "Minutes_Played": minutes,
                "Shots_on_Target": shots,
                "Pass_Accuracy_%": pass_acc,
                "Fatigue_Level": fatigue,
                "Assists": assists,
            }])
            yhat = float(reg_model.predict(X)[0])
            st.success(f"Predicted Goals: **{yhat:.2f}**")
        st.info("Tip: Retrain if you change the dataset or add more samples.")


# -----------------------------
# Predict Readiness (Classification)
# -----------------------------
with tabs[2]:
    st.subheader("🧠 Predict Player Readiness (Fatigue_Level)")
    st.caption("Inputs: Goals, Assists, Pass_Accuracy_%, Minutes_Played, Fouls, Shots_on_Target")

    colA, colB = st.columns([1, 1])

    with colA:
        g = st.number_input("Goals", min_value=0, max_value=10, value=1)
        a = st.number_input("Assists", min_value=0, max_value=10, value=0)
        p = st.number_input("Pass Accuracy (%)", min_value=0.0, max_value=100.0, value=85.0, step=0.1)
        m = st.number_input("Minutes Played", min_value=0, max_value=130, value=75)
        f = st.number_input("Fouls", min_value=0, max_value=10, value=1)
        s = st.number_input("Shots on Target", min_value=0, max_value=20, value=2)
        predict_fatigue = st.button("Predict Fatigue Level", type="primary", use_container_width=True)

    with colB:
        if predict_fatigue:
            X = pd.DataFrame([{
                "Goals": g,
                "Assists": a,
                "Pass_Accuracy_%": p,
                "Minutes_Played": m,
                "Fouls": f,
                "Shots_on_Target": s,
            }])

            pred = clf_model.predict(X)[0]
            st.success(f"Predicted Fatigue_Level: **{pred}**")

            if hasattr(clf_model.named_steps["model"], "predict_proba"):
                proba = clf_model.predict_proba(X)[0]
                labels = list(clf_model.named_steps["model"].classes_)
                prob_df = pd.DataFrame({"Class": labels, "Probability": proba}).sort_values("Probability", ascending=False)
                st.dataframe(prob_df, use_container_width=True)

            if pred == "High":
                st.warning("Recommendation: consider rest/rotation or reduced minutes.")
            elif pred == "Medium":
                st.info("Recommendation: playable, monitor workload.")
            else:
                st.success("Recommendation: likely ready.")


# -----------------------------
# Clustering
# -----------------------------
with tabs[3]:
    st.subheader("🧩 Player Clusters (K-Means)")
    st.caption("Clusters based on: Goals, Assists, Pass_Accuracy_%, Shots_on_Target")

    mode = st.radio("Mode", ["Select from dataset", "Manual input"], horizontal=True)

    if mode == "Select from dataset":
        player = st.selectbox("Player", options=sorted(df["Player"].astype(str).unique()))
        player_rows = df[df["Player"].astype(str) == str(player)].copy().sort_values("Match_Date", ascending=False)
        row = player_rows.iloc[0]

        X = pd.DataFrame([{
            "Goals": float(row["Goals"]) if pd.notna(row["Goals"]) else 0,
            "Assists": float(row["Assists"]) if pd.notna(row["Assists"]) else 0,
            "Pass_Accuracy_%": float(row["Pass_Accuracy_%"]) if pd.notna(row["Pass_Accuracy_%"]) else 0,
            "Shots_on_Target": float(row["Shots_on_Target"]) if pd.notna(row["Shots_on_Target"]) else 0,
        }])

        cluster = int(km_model.predict(X)[0])
        st.success(f"Cluster: **{cluster}**")
        st.dataframe(player_rows.head(10), use_container_width=True)

    else:
        col1, col2 = st.columns(2)
        with col1:
            g2 = st.number_input("Goals", min_value=0, max_value=10, value=1, key="km_g")
            a2 = st.number_input("Assists", min_value=0, max_value=10, value=0, key="km_a")
        with col2:
            p2 = st.number_input("Pass Accuracy (%)", min_value=0.0, max_value=100.0, value=85.0, step=0.1, key="km_p")
            s2 = st.number_input("Shots on Target", min_value=0, max_value=20, value=2, key="km_s")

        if st.button("Assign Cluster", type="primary"):
            X = pd.DataFrame([{
                "Goals": g2,
                "Assists": a2,
                "Pass_Accuracy_%": p2,
                "Shots_on_Target": s2,
            }])
            cluster = int(km_model.predict(X)[0])
            st.success(f"Cluster: **{cluster}**")


# -----------------------------
# Visual Analytics (PLOTS)
# -----------------------------
with tabs[4]:
    st.subheader("📈 Visual Analytics")

    # 1) Goals vs Shots on Target
    st.write("### Goals vs Shots on Target")
    fig1, ax1 = plt.subplots()
    ax1.scatter(df["Shots_on_Target"], df["Goals"])
    ax1.set_xlabel("Shots on Target")
    ax1.set_ylabel("Goals")
    ax1.set_title("Goals vs Shots on Target")
    st.pyplot(fig1)

    # 2) Pass Accuracy Distribution
    st.write("### Pass Accuracy Distribution")
    fig2, ax2 = plt.subplots()
    ax2.hist(df["Pass_Accuracy_%"].dropna(), bins=15)
    ax2.set_xlabel("Pass Accuracy (%)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of Pass Accuracy")
    st.pyplot(fig2)

    # 3) Fatigue Level Distribution
    st.write("### Fatigue Level Distribution")
    fatigue_counts = df["Fatigue_Level"].value_counts(dropna=False)
    fig3, ax3 = plt.subplots()
    ax3.bar(fatigue_counts.index.astype(str), fatigue_counts.values)
    ax3.set_xlabel("Fatigue Level")
    ax3.set_ylabel("Count")
    ax3.set_title("Fatigue Level Distribution")
    st.pyplot(fig3)

    # 4) Cluster Visualization (2D projection)
    st.write("### Player Clusters (K-Means Projection)")
    X_scaled = km_model.named_steps["preprocess"].transform(df[KM_FEATURES].copy())
    clusters = km_model.named_steps["model"].labels_

    fig4, ax4 = plt.subplots()
    ax4.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters)
    ax4.set_title("K-Means Player Clusters (2D projection)")
    ax4.set_xlabel("Feature 1 (scaled)")
    ax4.set_ylabel("Feature 2 (scaled)")
    st.pyplot(fig4)

    st.success("All report-style plots are included.")


# -----------------------------
# Data Explorer
# -----------------------------
with tabs[5]:
    st.subheader("📊 Data Explorer")

    c1, c2, c3 = st.columns(3)
    with c1:
        team = st.selectbox("Team", options=["All"] + sorted(df["Team"].astype(str).unique()))
    with c2:
        opp = st.selectbox("Opponent", options=["All"] + sorted(df["Opponent"].astype(str).unique()))
    with c3:
        fatigue = st.selectbox("Fatigue Level", options=["All", "Low", "Medium", "High"])

    view = df.copy()
    if team != "All":
        view = view[view["Team"].astype(str) == team]
    if opp != "All":
        view = view[view["Opponent"].astype(str) == opp]
    if fatigue != "All":
        view = view[view["Fatigue_Level"].astype(str) == fatigue]

    st.dataframe(view, use_container_width=True)

    st.download_button(
        "Download filtered CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="assistant_coach_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )


# -----------------------------
# Model Metrics
# -----------------------------
with tabs[6]:
    st.subheader("🛠️ Model Metrics")

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE (Goals)", f"{meta['regression']['rmse']:.2f}")
    c2.metric("MAE (Goals)", f"{meta['regression']['mae']:.2f}")
    c3.metric("R² (Goals)", f"{meta['regression']['r2']:.2f}")

    st.divider()
    st.metric("Fatigue Classification Accuracy", f"{meta['classification']['accuracy']*100:.1f}%")

    labels = meta["classification"]["labels"]
    cm = np.array(meta["classification"]["confusion_matrix"])
    cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
    st.write("Confusion Matrix (rows = actual, cols = predicted):")
    st.dataframe(cm_df, use_container_width=True)

    st.divider()
    st.metric("K-Means Silhouette Score", f"{meta['clustering']['silhouette']:.2f}")

    st.write("Saved model files:")
    st.code("\n".join(str(p) for p in model_paths().values()))
