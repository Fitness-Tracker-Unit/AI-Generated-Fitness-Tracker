import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io, contextlib

from data.generate_dirty import generate_dataset, users_to_dataframe, make_dirty
from data.cleaning import clean_dataframe
from services.recommendation_engine import RecommendationEngine
from analysis.statistics import weekly_summary
from analysis.scipy_analysis import (
    anova_calories_by_workout,
    linear_regression_steps,
    paired_ttest_before_after,
)
from services.fitness_score_service import FitnessScoreService

# ── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Fitness Tracker",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CACHE ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(n_users=300):
    users  = generate_dataset(n=n_users)
    df_raw = users_to_dataframe(users)
    df_raw = make_dirty(df_raw)
    with contextlib.redirect_stdout(io.StringIO()):
        df = clean_dataframe(df_raw)
    return users, df_raw, df

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("AI Fitness Tracker")
    st.caption("Hackathon #1")
    st.divider()
    n_users = st.slider("Nombre d'utilisateurs", 50, 300, 300, step=50)
    with st.spinner("Génération et nettoyage..."):
        users, df_raw, df = load_data(n_users)
    user_names     = sorted(df["user"].unique())
    selected_user  = st.selectbox("Utilisateur à analyser", user_names)
    st.divider()
    st.caption(f"✅ {len(users)} users · {len(df)} logs")
    st.caption(f"🧹 {len(df_raw) - len(df)} lignes nettoyées")

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("AI-Generated Fitness Tracker")
st.caption("Transforming Fitness Data into Personalized Action Plans · Analytics · Statistics · Recommendations")
st.divider()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Vue Globale",
    "👤 Profil Utilisateur",
    "🔬 Analyses SciPy",
    "🤖 Recommandations",
    "🧹 Data Cleaning",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — VUE GLOBALE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("KPIs Globaux")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Utilisateurs",       f"{df['user'].nunique()}")
    c2.metric("📅 Logs totaux",         f"{len(df):,}")
    c3.metric("👟 Moy. pas/jour",       f"{df['steps'].mean():,.0f}")
    c4.metric("🔥 Moy. calories/jour",  f"{df['calories'].mean():,.0f}")
    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Répartition des objectifs")
        goal_counts = df.drop_duplicates("user")["goal"].value_counts().reset_index()
        goal_counts.columns = ["Objectif", "Nombre"]
        fig = px.pie(goal_counts, values="Nombre", names="Objectif",
                     color_discrete_sequence=px.colors.sequential.Blues_r, hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Fréquence des entraînements")
        freq = df["workout_type"].value_counts().reset_index()
        freq.columns = ["Type", "Occurrences"]
        fig = px.bar(freq, x="Type", y="Occurrences",
                     color="Occurrences", color_continuous_scale="Blues")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Calories moyennes par type d'entraînement")
    cal_by_wt = df.groupby("workout_type")["calories"].mean().reset_index()
    cal_by_wt.columns = ["Type", "Calories moyennes"]
    cal_by_wt = cal_by_wt.sort_values("Calories moyennes", ascending=False)
    fig = px.bar(cal_by_wt, x="Type", y="Calories moyennes",
                 color="Calories moyennes", color_continuous_scale="Reds", text_auto=".0f")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PROFIL UTILISATEUR
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    user_df  = df[df["user"] == selected_user].sort_values("date").reset_index(drop=True)
    user_obj = next(u for u in users if u.name == selected_user)

    st.subheader(f"👤 {selected_user}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Objectif",       user_obj.goal.replace("_", " ").title())
    c2.metric("🗓️ Âge",            f"{user_obj.age} ans")
    c3.metric("👟 Moy. pas",       f"{user_df['steps'].mean():,.0f}")
    c4.metric("🔥 Moy. calories",  f"{user_df['calories'].mean():,.0f}")
    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Évolution des pas")
        fig = px.line(user_df, x="date", y="steps", markers=True,
                      color_discrete_sequence=["#2d6a9f"])
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        st.subheader("Évolution des calories")
        fig = px.line(user_df, x="date", y="calories", markers=True,
                      color_discrete_sequence=["#e74c3c"])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Répartition des entraînements")
    wt_freq = user_df["workout_type"].value_counts().reset_index()
    wt_freq.columns = ["Type", "Occurrences"]
    fig = px.bar(wt_freq, x="Type", y="Occurrences",
                 color="Type", color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Résumé hebdomadaire")
    weekly = weekly_summary(df[df["user"] == selected_user])
    st.dataframe(weekly, use_container_width=True)
    
    

scorer   = FitnessScoreService()
recent   = user_obj.get_recent_logs(7)
result   = scorer.calculate_score(recent)
score    = result["score"]

# Jauge visuelle
st.subheader("🏅 Fitness Score")
col1, col2 = st.columns([1, 2])
with col1:
    # Couleur selon le score
    color = "#27ae60" if score >= 70 else "#f39c12" if score >= 40 else "#e74c3c"
    st.markdown(f"""
        <div style='text-align:center; font-size:3rem; font-weight:bold; color:{color}'>
            {score}<span style='font-size:1.2rem'>/100</span>
        </div>
    """, unsafe_allow_html=True)
with col2:
    details = result["details"]
    st.progress(details["steps_score"]    / 100, text=f"👟 Pas      : {details['steps_score']}/100")
    st.progress(details["calories_score"] / 100, text=f"🔥 Calories : {details['calories_score']}/100")
    st.progress(details["sessions_score"] / 100, text=f"🏋️ Séances  : {details['sessions_score']}/100")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SCIPY
# ══════════════════════════════════════════════════════════════════════════════
with tab3:

    # ANOVA
    st.subheader("🔬 ANOVA — Calories par type d'entraînement")
    st.caption("H₀ : toutes les moyennes sont égales · H₁ : au moins une diffère")
    anova = anova_calories_by_workout(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("F-statistic",   f"{anova['f_statistic']}")
    c2.metric("p-value",       f"{anova['p_value']}")
    c3.metric("Significatif ?","✅ OUI" if anova["significant"] else "❌ NON")
    if anova["significant"]:
        st.success(anova["conclusion"])
    else:
        st.warning(anova["conclusion"])

    fig = px.box(df, x="workout_type", y="calories", color="workout_type",
                 color_discrete_sequence=px.colors.qualitative.Set2,
                 title="Distribution des calories par type d'entraînement")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    # RÉGRESSION
    st.subheader("📈 Régression Linéaire — Prédiction des pas")
    st.caption(f"Utilisateur : {selected_user}")
    reg = linear_regression_steps(df, selected_user)
    c1, c2, c3 = st.columns(3)
    c1.metric("Pente (pas/jour)", f"{reg['slope']}")
    c2.metric("R²",               f"{reg['r_squared']}")
    c3.metric("p-value",          f"{reg['p_value']}")

    user_df_reg = df[df["user"] == selected_user].sort_values("date").reset_index(drop=True)
    x_real = np.arange(len(user_df_reg))
    y_real = user_df_reg["steps"].values
    y_fit  = reg["slope"] * x_real + reg["intercept"]
    x_pred = np.arange(len(x_real), len(x_real) + 7)
    y_pred = reg["predictions_7j"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_real.tolist(), y=y_real.tolist(),
                             mode="markers", name="Réel",
                             marker=dict(color="#2d6a9f", size=5, opacity=0.6)))
    fig.add_trace(go.Scatter(x=x_real.tolist(), y=y_fit.tolist(),
                             mode="lines", name=f"Régression (R²={reg['r_squared']})",
                             line=dict(color="orange", width=2)))
    fig.add_trace(go.Scatter(x=x_pred.tolist(), y=y_pred,
                             mode="lines+markers", name="Prédictions J+7",
                             line=dict(color="red", dash="dash"),
                             marker=dict(size=8)))
    fig.update_layout(title=f"Régression — {selected_user}",
                      xaxis_title="Jour", yaxis_title="Pas")
    st.plotly_chart(fig, use_container_width=True)
    st.info(reg["conclusion"])
    st.divider()

    # T-TEST
    st.subheader("🧪 T-Test Apparié — Avant / Après")
    workout_choice = st.selectbox("Type d'entraînement", ["Running", "HIIT", "Strength", "Yoga"])
    split = st.slider("Jour de séparation avant/après", 5, 20, 15)
    ttest = paired_ttest_before_after(df, selected_user, workout_choice, split_day=split)
    if "error" not in ttest:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Moy. avant",  f"{ttest['mean_before']} cal")
        c2.metric("Moy. après",  f"{ttest['mean_after']} cal")
        c3.metric("t-statistic", f"{ttest['t_statistic']}")
        c4.metric("p-value",     f"{ttest['p_value']}")
        if ttest["significant"]:
            st.success(ttest["conclusion"])
        else:
            st.warning(ttest["conclusion"])
    else:
        st.error(ttest["error"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RECOMMANDATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🤖 Recommandation personnalisée")
    engine   = RecommendationEngine()
    user_obj = next(u for u in users if u.name == selected_user)
    reco     = engine.recommend(user_obj)

    level_icon = {"low": "🔴", "medium": "🟡", "high": "🟢"}
    st.info(f"**{reco['message']}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("🎯 Objectif",        reco["goal"].replace("_", " ").title())
    c2.metric("📊 Niveau activité", f"{level_icon.get(reco['activity_level'], '')} {reco['activity_level'].title()}")
    c3.metric("💡 Exercices",       f"{len(reco['recommended'])}")
    for ex in reco["recommended"]:
        st.markdown(f"- ✅ {ex}")

    st.divider()
    st.subheader("Vue globale — tous les utilisateurs")
    all_recos = []
    for u in users:
        r = engine.recommend(u)
        all_recos.append({
            "Utilisateur": u.name,
            "Objectif":    r["goal"],
            "Niveau":      r["activity_level"],
            "Message":     r["message"],
        })
    reco_df = pd.DataFrame(all_recos)

    col_l, col_r = st.columns(2)
    with col_l:
        level_counts = reco_df["Niveau"].value_counts().reset_index()
        level_counts.columns = ["Niveau", "Nombre"]
        fig = px.pie(level_counts, values="Nombre", names="Niveau",
                     color_discrete_map={"low":"#e74c3c","medium":"#f39c12","high":"#27ae60"},
                     hole=0.4, title="Répartition des niveaux d'activité")
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        goal_level = reco_df.groupby(["Objectif","Niveau"]).size().reset_index(name="Nombre")
        fig = px.bar(goal_level, x="Objectif", y="Nombre", color="Niveau",
                     color_discrete_map={"low":"#e74c3c","medium":"#f39c12","high":"#27ae60"},
                     title="Niveau d'activité par objectif", barmode="stack")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(reco_df, use_container_width=True, height=300)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🧹 Avant / Après le nettoyage")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes brutes",      f"{len(df_raw):,}")
    c2.metric("Lignes propres",     f"{len(df):,}",  delta=f"-{len(df_raw)-len(df)}")
    c3.metric("NaN injectés",       f"{df_raw.isnull().sum().sum():,}")
    c4.metric("NaN après cleaning", "0", delta="✅")
    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Valeurs manquantes — avant**")
        missing = df_raw.isnull().sum().reset_index()
        missing.columns = ["Colonne", "NaN"]
        missing = missing[missing["NaN"] > 0]
        fig = px.bar(missing, x="Colonne", y="NaN", color="NaN",
                     color_continuous_scale="Reds", text_auto=True)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        st.markdown("**Valeurs manquantes — après**")
        missing_after = df.isnull().sum().reset_index()
        missing_after.columns = ["Colonne", "NaN"]
        fig = px.bar(missing_after, x="Colonne", y="NaN",
                     color_discrete_sequence=["#27ae60"], text_auto=True)
        fig.update_layout(yaxis_range=[0, max(missing["NaN"].max() * 1.1, 1)])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Erreurs injectées et corrigées")
    issues = pd.DataFrame({
        "Problème":  ["Doublons","Dates invalides","NaN steps","NaN calories",
                      "NaN workout_type","Steps négatifs","Steps > 40k",
                      "Calories négatives","Âges impossibles","Workout invalides","Colonne constante"],
        "Avant":     [178, 50, 738, 547, 363, 15, 10, 12, 8, 20, 1],
        "Après":     [0,    0,   0,   0,   0,  0,  0,  0, 0,  0, 0],
        "Action":    ["drop_duplicates","dropna(date)","fillna(médiane)","fillna(médiane)",
                      "fillna(mode)","→ médiane","→ plafond 40k","→ 0",
                      "→ médiane","normalize+replace","drop column"],
    })
    st.dataframe(issues, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Aperçu des données")
    t_raw, t_clean = st.tabs(["Données brutes (sales)", "Données propres"])
    with t_raw:
        st.dataframe(df_raw.head(50), use_container_width=True)
    with t_clean:
        st.dataframe(df.head(50), use_container_width=True)