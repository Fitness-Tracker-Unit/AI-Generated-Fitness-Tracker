import random
import pandas as pd
import numpy as np
from datetime import date, timedelta
from faker import Faker
from models.user import User, UserLog

fake = Faker("fr_FR")

WORKOUT_TYPES = ["Running", "Yoga", "HIIT", "Strength", "Rest"]
GOALS = ["weight_loss", "strength", "endurance"]

CALORIES = {"Running":(450,60),"HIIT":(500,70),"Strength":(350,50),"Yoga":(200,40),"Rest":(80,20)}
STEPS    = {"Running":(8000,14000),"HIIT":(6000,10000),"Strength":(4000,7000),"Yoga":(2000,5000),"Rest":(500,3000)}


def generate_dataset(n=300):
    users = []
    for _ in range(n):
        user = User(
            name=fake.name(),
            age=random.randint(20, 55),
            city=fake.city(),
            goal=random.choice(GOALS),
        )
        for i in range(30):
            wt = random.choice(WORKOUT_TYPES)
            lo, hi = STEPS[wt]
            mu, sd = CALORIES[wt]
            user.add_log(UserLog(
                date=date.today() - timedelta(days=30 - i),
                steps=random.randint(lo, hi),
                calories=max(0, random.gauss(mu, sd)),
                workout_type=wt,
            ))
        users.append(user)
    return users


def users_to_dataframe(users):
    rows = []
    for u in users:
        for l in u.daily_logs:
            rows.append({
                "user": u.name, "age": u.age, "goal": u.goal,
                "city": u.city, "date": l.date,
                "steps": l.steps, "calories": l.calories,
                "workout_type": l.workout_type,
            })
    return pd.DataFrame(rows)


def make_dirty(df: pd.DataFrame, seed=42) -> pd.DataFrame:
    """
    Injecte volontairement des erreurs réalistes dans le DataFrame.
    Chaque saleté est documentée pour pouvoir montrer le avant/après.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)

    # ── 1. NaN dans steps (~8% des lignes)
    idx = rng.choice(n, size=int(n * 0.08), replace=False)
    df.loc[idx, "steps"] = np.nan

    # ── 2. NaN dans calories (~6% des lignes)
    idx = rng.choice(n, size=int(n * 0.06), replace=False)
    df.loc[idx, "calories"] = np.nan

    # ── 3. NaN dans workout_type (~4% des lignes)
    idx = rng.choice(n, size=int(n * 0.04), replace=False)
    df.loc[idx, "workout_type"] = np.nan

    # ── 4. Valeurs aberrantes (outliers) dans steps
    #       steps négatifs et steps impossibles (>50 000)
    idx_neg = rng.choice(n, size=15, replace=False)
    df.loc[idx_neg, "steps"] = rng.integers(-500, -1, size=15)

    idx_high = rng.choice(n, size=10, replace=False)
    df.loc[idx_high, "steps"] = rng.integers(55000, 80000, size=10)

    # ── 5. Âges impossibles
    idx_age = rng.choice(n, size=8, replace=False)
    df.loc[idx_age, "age"] = rng.choice([-3, 0, 150, 200], size=8)

    # ── 6. Doublons (~2% des lignes dupliquées)
    dup_idx = rng.choice(n, size=int(n * 0.02), replace=False)
    duplicates = df.iloc[dup_idx]
    df = pd.concat([df, duplicates], ignore_index=True)

    # ── 7. workout_type avec valeurs invalides
    idx_inv = rng.choice(len(df), size=20, replace=False)
    df.loc[idx_inv, "workout_type"] = rng.choice(
        ["running", "YOGA ", " hiit", "???", "N/A"], size=20
    )

    # ── 8. calories négatives
    idx_neg_cal = rng.choice(len(df), size=12, replace=False)
    df.loc[idx_neg_cal, "calories"] = rng.uniform(-200, -10, size=12)

    # ── 9. Colonne constante (inutile)
    df["source"] = "app_mobile"

    # ── 10. date en string au lieu de date object (type mixte)
    idx_str_date = rng.choice(len(df), size=50, replace=False)
    df["date"] = df["date"].astype(str)
    df.loc[idx_str_date, "date"] = "invalid_date"

    return df.reset_index(drop=True)