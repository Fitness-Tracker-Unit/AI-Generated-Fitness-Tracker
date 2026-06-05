from datetime import date
from dataclasses import dataclass, field
from typing import List


@dataclass
class UserLog:
    """Un journal quotidien d'activite."""
    date: date
    steps: int
    calories: float
    workout_type: str          # "Running", "Yoga", "HIIT", "Strength", "Rest"

    def __post_init__(self):
        if self.steps < 0:
            raise ValueError("Les pas ne peuvent pas etre negatifs.")
        if self.calories < 0:
            raise ValueError("Les calories ne peuvent pas etre negatives.")


@dataclass
class User:
    """Profil utilisateur — exactement ce que le sujet demande."""
    name: str
    age: int
    city: str = "Unknown"       # Champ optionnel pour les analyses geographiques
    goal: str                  # "weight_loss" | "strength" | "endurance"
    daily_logs: List[UserLog] = field(default_factory=list)

    def add_log(self, log: UserLog):
        self.daily_logs.append(log)

    def get_recent_logs(self, n_days: int = 7) -> List[UserLog]:
        """Retourne les n derniers jours de logs."""
        return self.daily_logs[-n_days:]

    def get_avg_steps(self, n_days: int = 7) -> float:
        logs = self.get_recent_logs(n_days)
        if not logs:
            return 0.0
        return sum(log.steps for log in logs) / len(logs)

    def get_avg_calories(self, n_days: int = 7) -> float:
        logs = self.get_recent_logs(n_days)
        if not logs:
            return 0.0
        return sum(log.calories for log in logs) / len(logs)

    def __repr__(self):
        return f"User(name={self.name}, age={self.age}, goal={self.goal})"