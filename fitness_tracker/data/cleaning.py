import pandas as pd
import numpy as np

VALID_WORKOUTS = ["Running", "Yoga", "HIIT", "Strength", "Rest"]


def clean_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline complet. Retourne le DataFrame propre.
    Affiche un rapport avant/après pour chaque étape.
    """
    df = df_raw.copy()
    _section("INSPECTION INITIALE")
    _inspect(df)

    _section("1. DOUBLONS")
    df = _remove_duplicates(df)

    _section("2. TYPES — date")
    df = _fix_date(df)

    _section("3. VALEURS MANQUANTES")
    df = _handle_missing(df)

    _section("4. VALEURS ABERRANTES")
    df = _fix_outliers(df)

    _section("5. WORKOUT_TYPE INVALIDES")
    df = _fix_workout_type(df)

    _section("6. COLONNES CONSTANTES")
    df = _drop_constant(df)

    _section("VALIDATION FINALE")
    _inspect(df)

    return df.reset_index(drop=True)


# ── Helpers internes ─────────────────────────────────────────────────

def _section(title):
    print(f"\n{'='*50}\n{title}\n{'='*50}")


def _inspect(df):
    print(f"Shape : {df.shape}")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("Valeurs manquantes : aucune ✓")
    else:
        print(f"Valeurs manquantes :\n{missing}")
    print(f"Doublons : {df.duplicated().sum()}")


def _remove_duplicates(df):
    avant = len(df)
    df = df.drop_duplicates()
    print(f"Supprimés : {avant - len(df)} doublons")
    return df


def _fix_date(df):
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    invalides = df["date"].isna().sum()
    print(f"Dates invalides remplacées par NaT : {invalides}")
    # Supprime les lignes sans date (non récupérables)
    df = df.dropna(subset=["date"])
    print(f"Lignes sans date supprimées : {invalides}")
    return df


def _handle_missing(df):
    # steps → médiane
    n = df["steps"].isna().sum()
    df["steps"] = df["steps"].fillna(df["steps"].median())
    print(f"steps    : {n} NaN → médiane ({df['steps'].median():.0f})")

    # calories → médiane
    n = df["calories"].isna().sum()
    df["calories"] = df["calories"].fillna(df["calories"].median())
    print(f"calories : {n} NaN → médiane")

    # workout_type → mode
    n = df["workout_type"].isna().sum()
    mode = df["workout_type"].mode()[0]
    df["workout_type"] = df["workout_type"].fillna(mode)
    print(f"workout  : {n} NaN → mode ('{mode}')")

    return df


def _fix_outliers(df):
    # Steps négatifs → médiane
    mask_neg = df["steps"] < 0
    df.loc[mask_neg, "steps"] = df["steps"].median()
    print(f"Steps négatifs corrigés  : {mask_neg.sum()}")

    # Steps > 40 000 → plafond
    mask_high = df["steps"] > 40000
    df.loc[mask_high, "steps"] = 40000
    print(f"Steps > 40 000 plafonnés : {mask_high.sum()}")

    # Calories négatives → 0
    mask_cal = df["calories"] < 0
    df.loc[mask_cal, "calories"] = 0
    print(f"Calories négatives → 0   : {mask_cal.sum()}")

    # Âges impossibles → médiane
    mask_age = ~df["age"].between(10, 100)
    median_age = df.loc[~mask_age, "age"].median()
    df.loc[mask_age, "age"] = median_age
    print(f"Âges impossibles corrigés: {mask_age.sum()}")

    return df


def _fix_workout_type(df):
    # Normalisation avec cas spécial HIIT
    def normalize(val):
        if pd.isna(val):
            return val
        v = str(val).strip()
        if v.upper() == "HIIT":
            return "HIIT"
        return v.capitalize()

    df["workout_type"] = df["workout_type"].apply(normalize)

    mask = ~df["workout_type"].isin(VALID_WORKOUTS)
    n = mask.sum()
    mode = df.loc[~mask, "workout_type"].mode()[0]
    df.loc[mask, "workout_type"] = mode
    print(f"Workout invalides remplacés par '{mode}' : {n}")
    return df


def _drop_constant(df):
    const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if const:
        df = df.drop(columns=const)
        print(f"Colonnes constantes supprimées : {const}")
    else:
        print("Aucune colonne constante.")
    return df


def save_clean(df: pd.DataFrame):
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed = os.path.join(base, "data", "processed")
    os.makedirs(processed, exist_ok=True)
    df[["user","age","goal","city"]].drop_duplicates().to_csv(
        os.path.join(processed, "users_clean.csv"), index=False)
    df[["user","date","steps","calories","workout_type"]].to_csv(
        os.path.join(processed, "activities_clean.csv"), index=False)
    print("CSV exportés → data/processed/")