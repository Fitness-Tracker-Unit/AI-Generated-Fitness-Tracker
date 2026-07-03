import sys, os
import io as _io
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
from services.fitness_score_service import FitnessScoreService
from analysis.statistics import weekly_summary
from analysis.scipy_analysis import (
    anova_calories_by_workout,
    linear_regression_steps,
    paired_ttest_before_after,
)

st.set_page_config(page_title="AI Fitness Tracker", page_icon="🏋️‍♂️", layout="wide")

if "dark" not in st.session_state:
    st.session_state.dark = True

DARK     = st.session_state.dark
BG       = "#0f1922"  if DARK else "#f0f4f8"
CARD     = "#1a2535"  if DARK else "#ffffff"
BORDER   = "#2d3f55"  if DARK else "#dde3ea"
TEXT     = "#ffffff"  if DARK else "#1a2535"
SUBTEXT  = "#90a4ae"  if DARK else "#607d8b"
SIDEBAR  = "#0f1922"  if DARK else "#e8edf2"
ACCENT   = "#4fc3f7"
PLOTLY_T = "plotly_dark" if DARK else "plotly_white"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Signika:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Signika', sans-serif;
    }

    .stApp {
        font-family: 'Signika', sans-serif;
    }

    h1, h2, h3, h4, h5, h6, p, span, div, label {
        font-family: 'Signika', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(f"""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
<style>
  body, .stApp {{ background:{BG}; color:{TEXT}; }}
  .kpi {{ background:{CARD}; border:1px solid {BORDER}; border-radius:14px;
          padding:20px; text-align:center; margin:4px; }}
  .kpi .ico {{ font-size:1.8rem; }}
  .kpi .val {{ font-size:1.8rem; font-weight:800; color:{ACCENT}; margin:6px 0 2px; }}
  .kpi .lbl {{ font-size:.75rem; color:{SUBTEXT}; letter-spacing:.5px; text-transform:uppercase; }}
  .sidebar-logo {{ text-align:center; padding:12px 0 4px; }}
  .sidebar-logo i {{ font-size:3rem; color:{ACCENT}; }}
  .sidebar-logo h2 {{ color:{TEXT}; margin:8px 0 2px; font-size:1.2rem; }}
  .sidebar-logo p  {{ color:{SUBTEXT}; font-size:.72rem; margin:0; }}
  section[data-testid="stSidebar"] {{ background:{SIDEBAR}; }}
  h1,h2,h3,h4 {{ color:{TEXT} !important; }}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data(n):
    users  = generate_dataset(n=n)
    df_raw = users_to_dataframe(users)
    df_raw = make_dirty(df_raw)
    with contextlib.redirect_stdout(io.StringIO()):
        df = clean_dataframe(df_raw)
    return users, df_raw, df

def kpi(col, icon, color, value, label, regular=False):
    style = "fa-regular" if regular else "fa-solid"
    col.markdown(f"""
    <div class='kpi'>
      <div class='ico' style='color:{color}'><i class='{style} {icon}'></i></div>
      <div class='val'>{value}</div>
      <div class='lbl'>{label}</div>
    </div>""", unsafe_allow_html=True)

def title(icon, text, color=ACCENT, regular=False):
    style = "fa-regular" if regular else "fa-solid"
    st.markdown(f"<h4><i class='{style} {icon}' style='color:{color}'></i>&nbsp; {text}</h4>",
                unsafe_allow_html=True)

# ── SIDEBAR
with st.sidebar:
    st.markdown(f"""
    <div class='sidebar-logo'>
      <i class='fa-solid fa-dumbbell'></i>
      <h2>AI Fitness Tracker</h2>
      <p>Hackathon #1 · Data Analytics</p>
    </div>""", unsafe_allow_html=True)
    st.divider()
    theme_label = "☀️ Mode clair" if DARK else "🌙 Mode sombre"
    if st.button(theme_label, use_container_width=True):
        st.session_state.dark = not st.session_state.dark
        st.rerun()
    st.divider()
    n_users = st.slider("Utilisateurs", 50, 300, 300, step=50)
    with st.spinner("Chargement..."):
        users, df_raw, df = load_data(n_users)
    user_names    = sorted(df["user"].unique())
    selected_user = st.selectbox("Analyser", user_names)
    st.divider()
    st.caption(f"✅ {len(users)} users · {len(df):,} logs")
    st.caption(f"🧹 {len(df_raw)-len(df)} lignes nettoyées")

# ── HEADER
st.markdown(f"""
<h1 style='margin-bottom:0;color:{TEXT}'>
  <i class='fa-solid fa-heart-pulse' style='color:#e74c3c'></i>&nbsp;
  AI-Generated Fitness Tracker
</h1>
<p style='color:{SUBTEXT};margin-top:4px'>
Transforming Fitness Data into Personalized Action Plans · Analytics · Statistics · AI Recommendations
</p>""", unsafe_allow_html=True)
st.divider()

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "À propos", "Vue Globale", "Profil Utilisateur",
    "Analyses SciPy", "Recommandations", "Data Cleaning"
])

# ══ TAB 0 — À PROPOS
with tab0:
    # Hero
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#1a2535,#0f3460);
                border-radius:16px;padding:36px 40px;margin-bottom:24px;
                border:1px solid {BORDER}'>
      <h1 style='color:#4fc3f7;margin:0 0 8px'>
        <i class='fa-solid fa-dumbbell'></i>&nbsp; AI-Generated Fitness Tracker
      </h1>
      <p style='color:{SUBTEXT};font-size:1rem;margin:0'>
        Dashboard analytique pour une plateforme intelligente de suivi fitness.
        Données synthétiques, data cleaning, analyses statistiques avancées
        et recommandations personnalisées par objectif.
      </p>
    </div>""", unsafe_allow_html=True)

    # Pipeline
    title("fa-diagram-project", "Pipeline du projet")
    steps = [
        ("fa-database",             "#4fc3f7", "Génération de données synthétiques",
         "300 utilisateurs générés avec Faker. 90 logs/user : pas, calories, type d'entraînement, ville."),
        ("fa-bug",                  "#ef5350", "Injection de données sales",
         "10 types d'erreurs : NaN, doublons, dates invalides, steps négatifs, calories négatives, âges impossibles, workout invalides, colonne constante."),
        ("fa-broom",                "#ffb74d", "Nettoyage avec Pandas",
         "Pipeline clean_dataframe() : doublons supprimés, NaN imputés (médiane/mode), outliers corrigés, types normalisés, export CSV propre."),
        ("fa-chart-bar",            "#81c784", "Analyse exploratoire",
         "Statistiques hebdomadaires groupby, fréquence des entraînements, calories moyennes par type, progression individuelle."),
        ("fa-flask",                "#ce93d8", "Analyses SciPy",
         "ANOVA (f_oneway) sur les calories par workout, régression linéaire (linregress) prédiction J+7, T-Test apparié (ttest_rel) avant/après."),
        ("fa-brain",                "#ffd700", "Moteur de recommandation",
         "Règles métier : niveau activité (low/medium/high) × objectif (weight_loss/strength/endurance) → plan d'entraînement personnalisé."),
        ("fa-trophy",               "#f39c12", "Fitness Score /100",
         "Score pondéré : 40% pas, 35% calories, 25% régularité. Normalisé par seuils cibles. Affiché avec jauge visuelle."),
        ("fa-chart-line",           "#4fc3f7", "Dashboard Streamlit",
         "Interface interactive Plotly + Streamlit. Dark/light mode, KPIs, 6 onglets, téléchargement CSV, sélection utilisateur."),
    ]
    for i, (icon, color, step_title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div style='display:flex;align-items:flex-start;gap:16px;
                    background:{CARD};border:1px solid {BORDER};
                    border-radius:12px;padding:16px 20px;margin-bottom:10px'>
          <div style='min-width:42px;height:42px;background:{color}22;border-radius:50%;
                      display:flex;align-items:center;justify-content:center;
                      font-size:1.1rem;color:{color};flex-shrink:0'>
            <i class='fa-solid {icon}'></i>
          </div>
          <div>
            <div style='color:{TEXT};font-weight:700;font-size:.95rem'>{i}. {step_title}</div>
            <div style='color:{SUBTEXT};font-size:.83rem;margin-top:3px'>{desc}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Stack technique
    title("fa-layer-group", "Stack technique", "#81c784")
    techs = [
        ("Python 3.14", "fa-brands fa-python",  "#4fc3f7"),
        ("Pandas",      "fa-solid fa-table",     "#81c784"),
        ("NumPy",       "fa-solid fa-calculator","#ffb74d"),
        ("SciPy",       "fa-solid fa-flask",     "#ce93d8"),
        ("Matplotlib",  "fa-solid fa-chart-line","#ef5350"),
        ("Seaborn",     "fa-solid fa-palette",   "#f39c12"),
        ("Streamlit",   "fa-solid fa-desktop",   "#ff4b4b"),
        ("Plotly",      "fa-solid fa-chart-bar", "#636efa"),
        ("Faker",       "fa-solid fa-user-secret","#90a4ae"),
    ]
    cols = st.columns(len(techs))
    for col, (name, icon, color) in zip(cols, techs):
        col.markdown(f"""
        <div style='background:{CARD};border:1px solid {BORDER};border-radius:10px;
                    padding:12px 8px;text-align:center'>
        <i class='{icon}' style='font-size:1.5rem;color:{color}'></i>
        <div style='font-size:.75rem;color:{SUBTEXT};margin-top:6px'>{name}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # KPIs projet
    title("fa-chart-pie", "Chiffres du projet", "#4fc3f7")
    c1,c2,c3,c4 = st.columns(4)
    kpi(c1,"fa-users",               "#4fc3f7", f"{df['user'].nunique()}",          "Utilisateurs")
    kpi(c2,"fa-calendar-days",       "#81c784", f"{len(df):,}",                     "Logs générés", regular=True)
    kpi(c3,"fa-triangle-exclamation","#ef5350", f"{df_raw.isnull().sum().sum():,}", "Erreurs injectées")
    kpi(c4,"fa-code",                "#ce93d8", "8",                                "Sprints Agile")

    st.divider()

    
    # Exigences du sujet
    title("fa-list-check", "Exigences du sujet couvertes", "#81c784")
    reqs = [
        "Profils utilisateurs (name, age, goal, daily_logs)",
        "Génération de workouts personnalisés selon objectif + activité",
        "ANOVA — scipy.stats.f_oneway",
        "Régression linéaire — scipy.stats.linregress",
        "T-Test apparié — scipy.stats.ttest_rel",
        "Pandas weekly summary (calories, steps, workouts)",
        "Line plots — steps et calories over time",
        "Bar plot — fréquence des types d'entraînement",
        "Data Cleaning — NaN, doublons, outliers, types",
    ]
    for req in reqs:
        st.markdown(
            f"<i class='fa-solid fa-circle-check' style='color:#81c784'></i>&nbsp; {req}",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Nos ajouts
    title("fa-star", "Nos ajouts — au-delà du sujet", "#ffd700")
    extras = [
        ("fa-trophy",        "#ffd700", "Fitness Score /100",
         "Score pondéré : 40% pas · 35% calories · 25% régularité des séances."),
        ("fa-chart-line",    "#4fc3f7", "Dashboard Streamlit interactif",
         "6 onglets, dark/light mode, KPIs, graphiques Plotly, téléchargement CSV."),
        ("fa-bug",           "#ef5350", "Injection volontaire d'erreurs",
         "10 types de saleté injectés dans les données pour démontrer le pipeline de nettoyage."),
        ("fa-moon",          "#ce93d8", "Dark / Light mode",
         "Changement de theme dark et light."),
    ]
    for icon, color, extra_title, desc in extras:
        st.markdown(f"""
        <div style='display:flex;align-items:flex-start;gap:14px;
                    background:{CARD};border:1px solid {BORDER};
                    border-radius:12px;padding:14px 18px;margin-bottom:8px'>
          <div style='min-width:38px;height:38px;background:{color}22;border-radius:50%;
                      display:flex;align-items:center;justify-content:center;
                      font-size:1rem;color:{color};flex-shrink:0'>
            <i class='fa-solid {icon}'></i>
          </div>
          <div>
            <div style='color:{TEXT};font-weight:700;font-size:.92rem'>{extra_title}</div>
            <div style='color:{SUBTEXT};font-size:.82rem;margin-top:2px'>{desc}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    st.markdown(
        f"""
        <div style="text-align:center; color:{SUBTEXT}; font-size:.85rem; font-family:'Signika', sans-serif;">
            AI-Generated Fitness Tracker · Hackathon #1 ·
            <i class="fa-solid fa-location-dot"></i> Abidjan, Côte d'Ivoire
            <br><br>
            Developed by
            <a href="https://carte-virtuelle-fahim.vercel.app/"
            target="_blank"
            style="font-weight:600; text-decoration:none;">
            Fahim Coulibaly
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ══ TAB 1 — VUE GLOBALE
with tab1:
    c1,c2,c3,c4 = st.columns(4)
    kpi(c1,"fa-users",             "#4fc3f7", df['user'].nunique(),            "Utilisateurs")
    kpi(c2,"fa-calendar-days",     "#81c784", f"{len(df):,}",                  "Logs totaux", regular=True)
    kpi(c3,"fa-shoe-prints",       "#ffb74d", f"{df['steps'].mean():,.0f}",    "Moy. pas / jour")
    kpi(c4,"fa-fire-flame-curved", "#ef5350", f"{df['calories'].mean():,.0f}", "Moy. calories / jour")
    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        title("fa-bullseye", "Répartition des objectifs")
        gc = df.drop_duplicates("user")["goal"].value_counts().reset_index()
        gc.columns = ["Objectif","Nombre"]
        st.plotly_chart(px.pie(gc, values="Nombre", names="Objectif",
            color_discrete_sequence=px.colors.sequential.Blues_r, hole=0.4,
            template=PLOTLY_T), use_container_width=True)
    with col_r:
        title("fa-chart-bar", "Fréquence des entraînements", "#81c784")
        fr = df["workout_type"].value_counts().reset_index()
        fr.columns = ["Type","Occurrences"]
        st.plotly_chart(px.bar(fr, x="Type", y="Occurrences",
            color="Occurrences", color_continuous_scale="Blues",
            template=PLOTLY_T).update_layout(coloraxis_showscale=False),
            use_container_width=True)

    title("fa-fire", "Calories moyennes par type d'entraînement", "#ef5350")
    cb = df.groupby("workout_type")["calories"].mean().reset_index()
    cb.columns = ["Type","Calories"]
    st.plotly_chart(px.bar(cb.sort_values("Calories", ascending=False),
        x="Type", y="Calories", color="Calories",
        color_continuous_scale="Reds", text_auto=".0f",
        template=PLOTLY_T).update_layout(coloraxis_showscale=False),
        use_container_width=True)

# ══ TAB 2 — PROFIL UTILISATEUR
with tab2:
    user_df  = df[df["user"]==selected_user].sort_values("date").reset_index(drop=True)
    user_obj = next(u for u in users if u.name==selected_user)

    st.markdown(f"<h4 style='color:{TEXT}'><i class='fa-regular fa-user' style='color:{ACCENT}'></i>&nbsp; {selected_user}</h4>",
                unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    kpi(c1,"fa-bullseye",          "#4fc3f7", user_obj.goal.replace("_"," ").title(), "Objectif")
    kpi(c2,"fa-cake-candles",      "#ce93d8", f"{user_obj.age} ans",                  "Âge")
    kpi(c3,"fa-shoe-prints",       "#ffb74d", f"{user_df['steps'].mean():,.0f}",      "Moy. pas")
    kpi(c4,"fa-fire-flame-curved", "#ef5350", f"{user_df['calories'].mean():,.0f}",   "Moy. calories")
    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        title("fa-person-running", "Évolution des pas")
        st.plotly_chart(px.line(user_df, x="date", y="steps", markers=True,
            color_discrete_sequence=["#4fc3f7"], template=PLOTLY_T), use_container_width=True)
    with col_r:
        title("fa-fire", "Évolution des calories", "#ef5350")
        st.plotly_chart(px.line(user_df, x="date", y="calories", markers=True,
            color_discrete_sequence=["#ef5350"], template=PLOTLY_T), use_container_width=True)

    title("fa-dumbbell", "Répartition des entraînements", "#81c784")
    wf = user_df["workout_type"].value_counts().reset_index()
    wf.columns = ["Type","Occurrences"]
    st.plotly_chart(px.bar(wf, x="Type", y="Occurrences", color="Type",
        color_discrete_sequence=px.colors.qualitative.Set2,
        template=PLOTLY_T), use_container_width=True)

    title("fa-table", "Résumé hebdomadaire", "#ffb74d")
    st.dataframe(weekly_summary(df[df["user"]==selected_user]), use_container_width=True)

    st.divider()
    title("fa-trophy", "Fitness Score", "#ffd700")
    scorer  = FitnessScoreService()
    recent  = user_obj.get_recent_logs(7)
    result  = scorer.calculate_score(recent)
    score   = result["score"]
    s_color = "#27ae60" if score>=70 else "#f39c12" if score>=40 else "#e74c3c"
    sc1, sc2 = st.columns([1,2])
    with sc1:
        st.markdown(f"""
        <div style='text-align:center;padding:24px;background:{CARD};
                    border:1px solid {BORDER};border-radius:14px'>
          <i class='fa-solid fa-medal' style='font-size:2.2rem;color:{s_color}'></i>
          <div style='font-size:3.2rem;font-weight:800;color:{s_color};margin:8px 0 0'>{score}</div>
          <div style='color:{SUBTEXT};font-size:.85rem'>/ 100</div>
        </div>""", unsafe_allow_html=True)
    with sc2:
        d = result["details"]
        st.progress(d["steps_score"]/100,    text=f"👟 Pas        {d['steps_score']}/100")
        st.progress(d["calories_score"]/100, text=f"🔥 Calories   {d['calories_score']}/100")
        st.progress(d["sessions_score"]/100, text=f"🏋️  Séances    {d['sessions_score']}/100")

# ══ TAB 3 — SCIPY
with tab3:
    title("fa-flask", "ANOVA — Calories par type d'entraînement", "#ce93d8")
    st.caption("H₀ : toutes les moyennes sont égales · H₁ : au moins une diffère")
    anova = anova_calories_by_workout(df)
    c1,c2,c3 = st.columns(3)
    kpi(c1,"fa-calculator","#4fc3f7", anova['f_statistic'], "F-statistic")
    kpi(c2,"fa-percent",   "#81c784", anova['p_value'],     "p-value")
    kpi(c3,
        "fa-circle-check" if anova["significant"] else "fa-circle-xmark",
        "#27ae60" if anova["significant"] else "#e74c3c",
        "OUI" if anova["significant"] else "NON", "Significatif")
    if anova["significant"]:
        st.success(anova["conclusion"])
    else:
        st.warning(anova["conclusion"])
    st.plotly_chart(px.box(df, x="workout_type", y="calories", color="workout_type",
        color_discrete_sequence=px.colors.qualitative.Set2,
        title="Distribution des calories par type",
        template=PLOTLY_T).update_layout(showlegend=False), use_container_width=True)
    st.divider()

    title("fa-chart-line", "Régression Linéaire — Prédiction des pas")
    reg = linear_regression_steps(df, selected_user)
    c1,c2,c3 = st.columns(3)
    kpi(c1,"fa-arrow-trend-up","#4fc3f7", reg['slope'],     "Pente (pas/jour)")
    kpi(c2,"fa-superscript",   "#81c784", reg['r_squared'], "R²")
    kpi(c3,"fa-percent",       "#ffb74d", reg['p_value'],   "p-value")
    u_reg = df[df["user"]==selected_user].sort_values("date").reset_index(drop=True)
    x_r   = np.arange(len(u_reg))
    y_r   = u_reg["steps"].values
    y_fit = reg["slope"]*x_r + reg["intercept"]
    x_p   = np.arange(len(x_r), len(x_r)+7)
    fig   = go.Figure()
    fig.add_trace(go.Scatter(x=x_r.tolist(), y=y_r.tolist(), mode="markers", name="Réel",
                             marker=dict(color="#4fc3f7", size=5, opacity=0.6)))
    fig.add_trace(go.Scatter(x=x_r.tolist(), y=y_fit.tolist(), mode="lines",
                             name=f"Régression (R²={reg['r_squared']})",
                             line=dict(color="orange", width=2)))
    fig.add_trace(go.Scatter(x=x_p.tolist(), y=reg["predictions_7j"], mode="lines+markers",
                             name="Prédictions J+7",
                             line=dict(color="#ef5350", dash="dash"), marker=dict(size=8)))
    fig.update_layout(title=f"Régression — {selected_user}",
                      xaxis_title="Jour", yaxis_title="Pas", template=PLOTLY_T)
    st.plotly_chart(fig, use_container_width=True)
    st.info(reg["conclusion"])
    st.divider()

    title("fa-vials", "T-Test Apparié — Avant / Après", "#ffb74d")
    workout_choice = st.selectbox("Type d'entraînement", ["Running","HIIT","Strength","Yoga"])
    ttest = paired_ttest_before_after(df, selected_user, workout_choice)
    if "error" not in ttest:
        st.caption(f"📊 {ttest['n_sessions']} séances analysées — coupure à la moitié")
        c1,c2,c3,c4 = st.columns(4)
        kpi(c1,"fa-clock-rotate-left","#4fc3f7", f"{ttest['mean_before']} cal","Moy. avant")
        kpi(c2,"fa-clock",            "#81c784", f"{ttest['mean_after']} cal", "Moy. après", regular=True)
        kpi(c3,"fa-t",                "#ffb74d", ttest['t_statistic'],         "t-statistic")
        kpi(c4,"fa-percent",          "#ce93d8", ttest['p_value'],             "p-value")
        if ttest["significant"]:
            st.success(ttest["conclusion"])
        else:
            st.warning(ttest["conclusion"])
    else:
        st.error(ttest["error"])
        st.info("💡 Pas assez de séances de ce type. Essaie un autre entraînement.")

# ══ TAB 4 — RECOMMANDATIONS
with tab4:
    title("fa-brain", "Recommandation personnalisée", "#ce93d8")
    engine   = RecommendationEngine()
    user_obj = next(u for u in users if u.name==selected_user)
    reco     = engine.recommend(user_obj)
    st.info(f"**{reco['message']}**")
    level_color = {"low":"#ef5350","medium":"#ffb74d","high":"#81c784"}
    c1,c2,c3 = st.columns(3)
    kpi(c1,"fa-bullseye",   "#4fc3f7", reco["goal"].replace("_"," ").title(), "Objectif")
    kpi(c2,"fa-signal",     level_color.get(reco["activity_level"],"#fff"),
        reco["activity_level"].title(), "Niveau activité")
    kpi(c3,"fa-list-check", "#81c784", len(reco["recommended"]), "Exercices")
    st.markdown("<br>", unsafe_allow_html=True)
    for ex in reco["recommended"]:
        st.markdown(
            f"<i class='fa-regular fa-circle-check' style='color:#81c784'></i>&nbsp; {ex}",
            unsafe_allow_html=True)
    st.divider()

    title("fa-users", "Vue globale — tous les utilisateurs")
    all_recos = [{"Utilisateur":u.name,"Objectif":r["goal"],
                  "Niveau":r["activity_level"],"Message":r["message"]}
                 for u in users for r in [engine.recommend(u)]]
    reco_df = pd.DataFrame(all_recos)
    col_l, col_r = st.columns(2)
    with col_l:
        lc = reco_df["Niveau"].value_counts().reset_index()
        lc.columns = ["Niveau","Nombre"]
        st.plotly_chart(px.pie(lc, values="Nombre", names="Niveau", hole=0.4,
            color_discrete_map={"low":"#ef5350","medium":"#ffb74d","high":"#81c784"},
            title="Répartition des niveaux", template=PLOTLY_T), use_container_width=True)
    with col_r:
        gl = reco_df.groupby(["Objectif","Niveau"]).size().reset_index(name="Nombre")
        st.plotly_chart(px.bar(gl, x="Objectif", y="Nombre", color="Niveau", barmode="stack",
            color_discrete_map={"low":"#ef5350","medium":"#ffb74d","high":"#81c784"},
            title="Niveau par objectif", template=PLOTLY_T), use_container_width=True)
    st.dataframe(reco_df, use_container_width=True, height=300)

# ══ TAB 5 — DATA CLEANING
with tab5:
    title("fa-broom", "Avant / Après le nettoyage", "#ffb74d")
    c1,c2,c3,c4 = st.columns(4)
    kpi(c1,"fa-database",             "#ef5350", f"{len(df_raw):,}",                  "Lignes brutes")
    kpi(c2,"fa-circle-check",         "#81c784", f"{len(df):,}",                      "Lignes propres")
    kpi(c3,"fa-triangle-exclamation", "#ffb74d", f"{df_raw.isnull().sum().sum():,}",  "NaN injectés")
    kpi(c4,"fa-shield-halved",        "#4fc3f7", "0",                                 "NaN restants")
    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        title("fa-bug", "Valeurs manquantes — avant", "#ef5350")
        missing = df_raw.isnull().sum().reset_index()
        missing.columns = ["Colonne","NaN"]
        missing = missing[missing["NaN"]>0]
        st.plotly_chart(px.bar(missing, x="Colonne", y="NaN", color="NaN",
            color_continuous_scale="Reds", text_auto=True,
            template=PLOTLY_T).update_layout(coloraxis_showscale=False),
            use_container_width=True)
    with col_r:
        title("fa-circle-check", "Valeurs manquantes — après", "#81c784")
        ma = df.isnull().sum().reset_index()
        ma.columns = ["Colonne","NaN"]
        st.plotly_chart(px.bar(ma, x="Colonne", y="NaN",
            color_discrete_sequence=["#81c784"], text_auto=True,
            template=PLOTLY_T).update_layout(
            yaxis_range=[0, max(missing["NaN"].max()*1.1,1)]),
            use_container_width=True)

    title("fa-table-list", "Erreurs injectées et corrigées")
    st.dataframe(pd.DataFrame({
        "Problème": ["Doublons","Dates invalides","NaN steps","NaN calories","NaN workout_type",
                     "Steps négatifs","Steps > 40k","Calories négatives","Âges impossibles",
                     "Workout invalides","Colonne constante"],
        "Avant":    [178,50,738,547,363,15,10,12,8,20,1],
        "Après":    [0]*11,
        "Action":   ["drop_duplicates","dropna(date)","fillna(médiane)","fillna(médiane)",
                     "fillna(mode)","→ médiane","→ plafond 40k","→ 0",
                     "→ médiane","normalize+replace","drop column"],
    }), use_container_width=True, hide_index=True)

    st.divider()
    title("fa-eye", "Aperçu des données", "#ce93d8", regular=True)
    t_raw, t_clean = st.tabs(["Données brutes (sales)", "Données propres"])

    def to_excel_bytes(dataframe):
        buf = _io.BytesIO()
        dataframe.to_excel(buf, index=False, engine="openpyxl")
        return buf.getvalue()


    with t_raw:
        st.dataframe(df_raw.head(100), use_container_width=True)
        st.download_button("⬇️ Telecharger CSV sale",
                           df_raw.to_csv(index=False).encode(),
                           "activities_dirty.csv", "text/csv")
        st.download_button("⬇️ Telecharger Excel sale",
                           to_excel_bytes(df_raw),
                           "activities_dirty.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with t_clean:
        st.dataframe(df.head(100), use_container_width=True)
        st.download_button("⬇️ Telecharger CSV propre",
                           df.to_csv(index=False).encode(),
                           "activities_clean.csv", "text/csv")
        st.download_button("⬇️ Telecharger Excel propre",
                           to_excel_bytes(df),
                           "activities_clean.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")