import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



sns.set_theme(style="whitegrid")


def plot_steps_over_time(df: pd.DataFrame, user_name: str):
    """Line plot : evolution des pas dans le temps."""
    user_df = df[df["user"] == user_name].sort_values("date")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(user_df["date"], user_df["steps"], marker="o", linewidth=1.5)
    ax.set_title(f"Evolution des pas — {user_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Pas")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"outputs/{user_name}_steps.png", dpi=120)
    plt.show()


def plot_calories_over_time(df: pd.DataFrame, user_name: str):
    """Line plot : evolution des calories brulees."""
    user_df = df[df["user"] == user_name].sort_values("date")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(user_df["date"], user_df["calories"],
            marker="o", color="tomato", linewidth=1.5)
    ax.set_title(f"Calories brulees — {user_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Calories")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"outputs/{user_name}_calories.png", dpi=120)
    plt.show()


def plot_workout_frequency(df: pd.DataFrame, user_name: str = None):
    """Bar plot : frequence des types d'entrainement."""
    data = df if user_name is None else df[df["user"] == user_name]
    freq = data["workout_type"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=freq.index, y=freq.values, ax=ax, hue=freq.index, palette="viridis", legend=False)
    title = f"Frequence des entrainements — {user_name}" if user_name else "Frequence globale"
    ax.set_title(title)
    ax.set_xlabel("Type")
    ax.set_ylabel("Occurrences")
    plt.tight_layout()
    plt.savefig(f"outputs/freq_{user_name or 'global'}.png", dpi=120)
    plt.show()
    
def plot_regression_steps(df, user_name, reg_result):
    """Line plot : données réelles + droite de régression + prédictions."""
    import numpy as np
    u = df[df["user"] == user_name].sort_values("date").reset_index(drop=True)

    x_real = np.arange(len(u))
    y_real = u["steps"].values
    y_fit  = reg_result["slope"] * x_real + reg_result["intercept"]

    # 7 prédictions futures
    x_pred = np.arange(len(u), len(u) + 7)
    y_pred = reg_result["predictions_7j"]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(x_real, y_real, label="Réel", alpha=0.6, s=20)
    ax.plot(x_real, y_fit, color="orange", label=f"Régression (R²={reg_result['r_squared']})")
    ax.plot(x_pred, y_pred, "r--o", label="Prédictions J+7")
    ax.set_title(f"Régression linéaire — {user_name}")
    ax.set_xlabel("Jour"); ax.set_ylabel("Pas")
    ax.legend(); plt.tight_layout()
    plt.savefig(f"outputs/{user_name}_regression.png", dpi=120)
    plt.show()