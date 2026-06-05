import pandas as pd


def weekly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le total calories, pas et seances par semaine et par user.
    Exactement ce que demande le point 4 du sujet.
    """
    df = df.copy()
    df["week"] = pd.to_datetime(df["date"]).dt.isocalendar().week

    summary = (
        df.groupby(["user", "week"])
        .agg(
            total_steps=("steps", "sum"),
            total_calories=("calories", "sum"),
            nb_workouts=("workout_type", lambda x: (x != "Rest").sum()),
            avg_steps=("steps", "mean"),
            avg_calories=("calories", "mean"),
        )
        .reset_index()
    )
    return summary


def workout_frequency(df: pd.DataFrame) -> pd.Series:
    """Frequence de chaque type d'entrainement."""
    return df["workout_type"].value_counts()


def user_progression(df: pd.DataFrame, user_name: str) -> pd.DataFrame:
    """Retourne les logs tri\u00e9s par date pour un utilisateur."""
    return (
        df[df["user"] == user_name]
        .sort_values("date")
        .reset_index(drop=True)
    )