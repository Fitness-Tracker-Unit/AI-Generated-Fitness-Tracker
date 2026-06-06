class FitnessScoreService:

    TARGET_STEPS    = 10_000
    TARGET_CALORIES = 500
    TARGET_SESSIONS = 5

    W_STEPS    = 0.40
    W_CALORIES = 0.35
    W_SESSIONS = 0.25

    def calculate_score(self, activities: list) -> dict:
        if not activities:
            return {"score": 0, "details": {}}

        avg_steps    = sum(a.steps    for a in activities) / len(activities)
        avg_calories = sum(a.calories for a in activities) / len(activities)
        nb_sessions  = len([a for a in activities if a.workout_type != "Rest"])

        s_steps    = min(avg_steps    / self.TARGET_STEPS,    1.0)
        s_calories = min(avg_calories / self.TARGET_CALORIES, 1.0)
        s_sessions = min(nb_sessions  / self.TARGET_SESSIONS, 1.0)

        final = (s_steps * self.W_STEPS +
                 s_calories * self.W_CALORIES +
                 s_sessions * self.W_SESSIONS) * 100

        return {
            "score": round(final, 1),
            "details": {
                "steps_score":    round(s_steps    * 100, 1),
                "calories_score": round(s_calories * 100, 1),
                "sessions_score": round(s_sessions * 100, 1),
                "avg_steps":      round(avg_steps),
                "avg_calories":   round(avg_calories),
                "nb_sessions":    nb_sessions,
            }
        }