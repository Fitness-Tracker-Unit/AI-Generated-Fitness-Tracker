import os
import sys
from data.generate_dirty import generate_dataset, users_to_dataframe, make_dirty, save_dirty
from data.cleaning import clean_dataframe, save_clean
from services.recommendation_engine import RecommendationEngine
from services.fitness_score_service import FitnessScoreService
from analysis.statistics import weekly_summary
from analysis.scipy_analysis import (
    anova_calories_by_workout,
    linear_regression_steps,
    paired_ttest_before_after,
)
from visualizations.charts import (
    plot_steps_over_time,
    plot_calories_over_time,
    plot_workout_frequency,
    plot_regression_steps,
)

os.makedirs("outputs", exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ── 1. Donnees ────────────────────────────────────────────────────

print("-" * 20)
print("GENERATION DES DONNEES")
print("-" * 20)
users  = generate_dataset(n=300)
df_raw = users_to_dataframe(users)
df_raw = make_dirty(df_raw)          # ← injection des erreurs
save_dirty(df_raw)   # ← ajoute cette ligne
print(f"Brut (sale) : {df_raw.shape}")
print(f"NaN total   : {df_raw.isnull().sum().sum()}")

print("-" * 20)
print("NETTOYAGE")
print("-" * 20)
df = clean_dataframe(df_raw)
save_clean(df)
print(f"\nPropre : {df.shape}")

# ── tout le reste du main.py reste identique ──
# (recommandations, stats, scipy, visualisations)
# il utilise df (propre) dès ici


# ── 2. Recommandations ───────────────────────────────────────────
print("-" * 20)
print("RECOMMANDATIONS PERSONNALISEES")
print("-" * 20)
engine = RecommendationEngine()
for user in users:
    reco = engine.recommend(user)
    print(f"[{user.name}] Objectif: {user.goal}")
    print(f"  → {reco['message']}")
    print(f"  → Niveau activite: {reco['activity_level']}")
    print()


# ──  Fitness score ───────────────────────────────
scorer = FitnessScoreService()
print("\n=== FITNESS SCORES ===")
for user in users[:5]:  # les 5 premiers pour la démo
    recent = user.get_recent_logs(7)
    result = scorer.calculate_score(recent)
    print(f"{user.name} → {result['score']}/100")
    
    
# ── 3. Statistiques hebdomadaires ───────────────────────────────
print("-" * 20)
print("RESUME HEBDOMADAIRE (Pandas)")
print("-" * 20)
summary = weekly_summary(df)
print(summary.to_string(index=False))
print()

# ── 4. SciPy ─────────────────────────────────────────────────────
print("-" * 20)
print("ANALYSES SCIPY")
print("-" * 20)

print("\n--- ANOVA : calories par type d'entrainement ---")
anova = anova_calories_by_workout(df)
print(f"F = {anova['f_statistic']}  |  p = {anova['p_value']}")
print(anova["conclusion"])

print("\n--- Regression lineaire : prediction des pas ---")
first_user = users[0].name
reg = linear_regression_steps(df, first_user)
print(f"Utilisateur : {reg['user']}")
print(f"Pente : {reg['slope']} pas/jour  |  R² = {reg['r_squared']}")
print(f"Predictions 7j : {reg['predictions_7j']}")
print(reg["conclusion"])

print("\n--- T-test apparie : avant/apres Running ---")
ttest = paired_ttest_before_after(df, first_user, "Running", split_day=15)
if "error" not in ttest:
    print(f"Moyenne avant : {ttest['mean_before']} cal")
    print(f"Moyenne apres : {ttest['mean_after']} cal")
    print(ttest["conclusion"])
else:
    print(ttest["error"])

# ── 5. Visualisations ────────────────────────────────────────────
print("\n" + "-" * 20)
print("VISUALISATIONS")
print("-" * 20)
plot_steps_over_time(df, first_user)
plot_calories_over_time(df, first_user)
plot_workout_frequency(df, first_user)
plot_regression_steps(df, first_user, reg)
print("Graphiques sauvegardes dans outputs/")