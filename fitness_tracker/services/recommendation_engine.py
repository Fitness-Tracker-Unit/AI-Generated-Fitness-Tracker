from models.user import User, UserLog
from typing import List


class RecommendationEngine:
    """
    Genere un plan d'entrainement personnalise
    selon l'objectif et le niveau d'activite recent.
    """

    RULES = {
        "weight_loss": {
            "low":    ["Marche rapide 30min", "Cardio leger 15min"],
            "medium": ["Running 30min", "Velo 30min"],
            "high":   ["HIIT 45min", "Running 45min"],
        },
        "strength": {
            "low":    ["Etirements 20min", "Yoga 30min"],
            "medium": ["Musculation legere 30min", "Bodyweight 25min"],
            "high":   ["Musculation lourde 45min", "CrossFit 30min"],
        },
        "endurance": {
            "low":    ["Marche 40min", "Velo doux 20min"],
            "medium": ["Running 40min", "Natation 30min"],
            "high":   ["Running 60min", "HIIT 45min"],
        },
    }

    def _get_activity_level(self, logs: List[UserLog]) -> str:
        if not logs:
            return "low"
        avg = sum(log.steps for log in logs) / len(logs)
        if avg < 5000:
            return "low"
        elif avg < 9000:
            return "medium"
        return "high"

    def recommend(self, user: User) -> dict:
        """
        Retourne une recommandation textuelle
        exactement comme demande dans le sujet.
        """
        recent = user.get_recent_logs(7)
        level  = self._get_activity_level(recent)
        goal   = user.goal if user.goal in self.RULES else "endurance"
        plans  = self.RULES[goal][level]

        return {
            "user":             user.name,
            "goal":             goal,
            "activity_level":   level,
            "recommended":      plans,
            # Format exact demande dans le sujet
            "message": (
                f"Based on your activity, try a {plans[0]} today."
            ),
        }