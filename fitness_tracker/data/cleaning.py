# data/cleaning.py
import pandas as pd

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage du DataFrame brut.
    Retourne un DataFrame propre prêt pour l'analyse.
    """
    df = df.copy()

    # 1. Suppression des doublons
    avant = len(df)
    df = df.drop_duplicates()
    print(f"  Doublons supprimés : {avant - len(df)}")

    # 2. Valeurs manquantes
    print(f"  Valeurs manquantes avant :\n{df.isnull().sum()}")
    df["steps"].fillna(df["steps"].median(), inplace=True)
    df["calories"].fillna(df["calories"].median(), inplace=True)
    df["workout_type"].fillna("Rest", inplace=True)
    print(f"  Valeurs manquantes après : {df.isnull().sum().sum()}")

    # 3. Validation des colonnes
    df = df[df["steps"] >= 0]
    df = df[df["calories"] >= 0]
    df = df[df["age"].between(10, 100)]
    valid_workouts = ["Running", "Yoga", "HIIT", "Strength", "Rest"]
    df = df[df["workout_type"].isin(valid_workouts)]

    # 4. Types corrects
    df["date"] = pd.to_datetime(df["date"])
    df["steps"] = df["steps"].astype(int)
    df["calories"] = df["calories"].astype(float)

    return df.reset_index(drop=True)


def save_clean(df: pd.DataFrame):
    """Export vers processed/ — remplit les CSV vides."""
    df[["user","age","goal","city"]].drop_duplicates().to_csv(
        "data/processed/users_clean.csv", index=False)
    df[["user","date","steps","calories","workout_type"]].to_csv(
        "data/processed/activities_clean.csv", index=False)
    print("  CSV exportés dans data/processed/")