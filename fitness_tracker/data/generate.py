import random
import pandas as pd
from datetime import date, timedelta
from models.user import User, UserLog
from faker import Faker


WORKOUT_TYPES = ["Running", "Yoga", "HIIT", "Strength", "Rest"]
GOALS = ["weight_loss", "strength", "endurance"]
fake = Faker("fr_FR")

def generate_user(name: str, age: int, city: str, goal: str, n_days: int = 30) -> User:
    """Genere un utilisateur avec n_days de logs simules."""
    user = User(name=name, age=age, city=city, goal=goal)

    for i in range(n_days):
        workout = random.choice(WORKOUT_TYPES)

        # Calories selon le type — volontairement separees pour l'ANOVA
        calories_map = {
            "Running":  random.gauss(450, 60),
            "HIIT":     random.gauss(500, 70),
            "Strength": random.gauss(350, 50),
            "Yoga":     random.gauss(200, 40),
            "Rest":     random.gauss(80,  20),
        }
        steps_map = {
            "Running":  random.randint(8000, 14000),
            "HIIT":     random.randint(6000, 10000),
            "Strength": random.randint(4000, 7000),
            "Yoga":     random.randint(2000, 5000),
            "Rest":     random.randint(500,  3000),
        }

        log = UserLog(
            date=date.today() - timedelta(days=n_days - i),
            steps=steps_map[workout],
            calories=max(0, calories_map[workout]),
            workout_type=workout,
        )
        user.add_log(log)

    return user


def generate_dataset(n_users=300):
    users = []

    for _ in range(n_users):

        user = generate_user(
            name=fake.name(),
            age=random.randint(20, 55),
            goal=random.choice(GOALS),
            city=fake.city(),
            n_days=30
        )

        users.append(user)

    return users


def users_to_dataframe(users: list) -> pd.DataFrame:
    """Convertit la liste d'utilisateurs en DataFrame Pandas."""
    rows = []
    for user in users:
        for log in user.daily_logs:
            rows.append({
                "user":         user.name,
                "age":          user.age,
                "goal":         user.goal,
                "city":         user.city,
                "date":         log.date,
                "steps":        log.steps,
                "calories":     log.calories,
                "workout_type": log.workout_type,
            })
    return pd.DataFrame(rows)



# if __name__ == "__main__":

#     users = generate_dataset(300)

#     df = users_to_dataframe(users)

#     print(df.head())

#     print(f"\nShape : {df.shape}")

#     df.to_csv("data/raw/activities.csv", index=False)

#     print("\nDataset généré avec succès.")