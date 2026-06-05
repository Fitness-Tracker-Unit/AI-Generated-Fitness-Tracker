import pandas as pd
import numpy as np
from scipy import stats


# ─── 1. ANOVA ────────────────────────────────────────────────────────────────

def anova_calories_by_workout(df: pd.DataFrame) -> dict:
    """
    Test ANOVA : est-ce que les types d'entrainement brulent
    des quantites de calories significativement differentes ?

    H0 : toutes les moyennes sont egales.
    H1 : au moins une moyenne differe.
    """
    groups = [
        df[df["workout_type"] == wt]["calories"].values
        for wt in df["workout_type"].unique()
    ]
    f_stat, p_value = stats.f_oneway(*groups)

    return {
        "test":        "ANOVA one-way",
        "f_statistic": round(f_stat, 4),
        "p_value":     round(p_value, 6),
        "significant": p_value < 0.05,
        "conclusion": (
            "Les types d'entrainement ont des effets SIGNIFICATIVEMENT differents "
            "sur les calories brulees (p < 0.05)."
            if p_value < 0.05 else
            "Pas de difference significative entre les types d'entrainement."
        ),
    }


# ─── 2. REGRESSION LINEAIRE ──────────────────────────────────────────────────

def linear_regression_steps(df: pd.DataFrame, user_name: str) -> dict:
    """
    Regression lineaire : predit les pas futurs
    a partir du trend des derniers jours.
    """
    user_df = (
        df[df["user"] == user_name]
        .sort_values("date")
        .reset_index(drop=True)
    )

    x = np.arange(len(user_df))          # jours 0, 1, 2...
    y = user_df["steps"].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Prediction pour les 7 prochains jours
    next_days = np.arange(len(x), len(x) + 7)
    predictions = slope * next_days + intercept

    return {
        "test":          "Regression lineaire",
        "user":          user_name,
        "slope":         round(slope, 2),
        "intercept":     round(intercept, 2),
        "r_squared":     round(r_value ** 2, 4),
        "p_value":       round(p_value, 6),
        "predictions_7j": [round(p) for p in predictions],
        "conclusion": (
            f"Tendance {'croissante' if slope > 0 else 'decroissante'} "
            f"de {abs(round(slope, 1))} pas/jour. "
            f"R² = {round(r_value**2, 3)} "
            f"({'bon fit' if r_value**2 > 0.5 else 'fit faible'})."
        ),
    }


# ─── 3. T-TEST APPARIE ───────────────────────────────────────────────────────

def paired_ttest_before_after(df: pd.DataFrame,
                               user_name: str,
                               workout_type: str,
                               split_day: int = 15) -> dict:
    """
    T-test apparie : l'entrainement a-t-il significativement
    change les calories brulees avant / apres un regime ?

    split_day : le jour qui separe "avant" et "apres".
    """
    user_df = (
        df[(df["user"] == user_name) & (df["workout_type"] == workout_type)]
        .sort_values("date")
        .reset_index(drop=True)
    )

    before = user_df.iloc[:split_day]["calories"].values
    after  = user_df.iloc[split_day:]["calories"].values

    # Egalise les longueurs pour le test apparie
    min_len = min(len(before), len(after))
    if min_len < 2:
        return {"error": "Pas assez de donnees pour ce test."}

    before, after = before[:min_len], after[:min_len]
    t_stat, p_value = stats.ttest_rel(before, after)

    return {
        "test":          "T-test apparie",
        "user":          user_name,
        "workout":       workout_type,
        "mean_before":   round(np.mean(before), 2),
        "mean_after":    round(np.mean(after), 2),
        "t_statistic":   round(t_stat, 4),
        "p_value":       round(p_value, 6),
        "significant":   p_value < 0.05,
        "conclusion": (
            f"Le programme '{workout_type}' a produit un changement "
            f"SIGNIFICATIF sur les calories (p < 0.05)."
            if p_value < 0.05 else
            f"Pas de difference significative avant/apres '{workout_type}'."
        ),
    }