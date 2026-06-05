import os
from data.generate import generate_dataset, users_to_dataframe
from data.cleaning import clean_dataframe, save_clean
from services.recommendation_engine import RecommendationEngine
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

# ── 1. Donnees ────────────────────────────────────────────────────
print("=" * 50)
print("GENERATION DES DONNEES")
print("=" * 50)
users = generate_dataset(n_users=300)
df    = users_to_dataframe(users)
print(f"{len(users)} utilisateurs, {len(df)} logs generes.\n")

# ── 2. Recommandations ───────────────────────────────────────────
print("=" * 50)
print("RECOMMANDATIONS PERSONNALISEES")
print("=" * 50)
engine = RecommendationEngine()
for user in users:
    reco = engine.recommend(user)
    print(f"[{user.name}] Objectif: {user.goal}")
    print(f"  → {reco['message']}")
    print(f"  → Niveau activite: {reco['activity_level']}")
    print()

# ── 3. Statistiques hebdomadaires ───────────────────────────────
print("=" * 50)
print("RESUME HEBDOMADAIRE (Pandas)")
print("=" * 50)
summary = weekly_summary(df)
print(summary.to_string(index=False))
print()

# ── 4. SciPy ─────────────────────────────────────────────────────
print("=" * 50)
print("ANALYSES SCIPY")
print("=" * 50)

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
print("\n" + "=" * 50)
print("VISUALISATIONS")
print("=" * 50)
plot_steps_over_time(df, first_user)
plot_calories_over_time(df, first_user)
plot_workout_frequency(df, first_user)
plot_regression_steps(df, first_user, reg)
print("Graphiques sauvegardes dans outputs/")