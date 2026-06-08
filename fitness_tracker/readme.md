# AI Fitness Tracker

## Overview

AI Fitness Tracker is a Python-based data analytics project that simulates a real-world fitness tracking platform.

The project demonstrates:

* Data generation
* Data cleaning
* Statistical analysis
* Personalized recommendations
* Fitness scoring
* Data visualization

It is designed as a portfolio project for Data Analyst, Data Scientist, and Data Engineer roles.

---

## Features

### Synthetic Data Generation

Generate realistic fitness activity logs for hundreds of users:

* Steps
* Calories burned
* Workout type
* Fitness goals
* Activity history

---

### Data Quality Simulation

The project intentionally injects common data issues:

* Missing values
* Duplicates
* Invalid categories
* Negative values
* Outliers
* Corrupted dates

---

### Data Cleaning Pipeline

Automated cleaning process:

* Duplicate removal
* Missing value handling
* Outlier correction
* Date validation
* Category normalization

---

### Personalized Recommendation Engine

Recommendations are generated according to:

* User goal
* Recent activity level

Supported goals:

* Weight Loss
* Strength
* Endurance

---

### Fitness Score

Each user receives a score from 0 to 100 based on:

| Metric   | Weight |
| -------- | ------ |
| Steps    | 40%    |
| Calories | 35%    |
| Sessions | 25%    |

---

### Statistical Analysis

Implemented statistical methods:

#### ANOVA

Compare calories burned across workout types.

#### Linear Regression

Predict future step counts.

#### Paired T-Test

Measure changes before and after a training period.

---

### Visualization

Automatically generates:

* Step trends
* Calories trends
* Workout frequency charts
* Regression plots

---

## Project Structure

```text
fitness_tracker/
│
├── analysis/
│   ├── statistics.py
│   ├── scipy_analysis.py
│   └── forecasting.py
│
├── data/
│   ├── generate.py
│   ├── generate_dirty.py
│   └── cleaning.py
│
├── models/
│   ├── user.py
│   ├── activity.py
│   └── workout_plan.py
│
├── services/
│   ├── recommendation_engine.py
│   ├── fitness_score_service.py
│   └── alert_service.py
│
├── visualizations/
│   ├── charts.py
│   └── dashboard.py
│
├── repositories/
│
├── outputs/
│
├── tests/
│
└── main.py
```

---

## Installation

```bash
git clone <repository-url>

cd fitness_tracker

pip install -r requirements.txt
```

---

## Run

```bash
python main.py
```

---

## Generated Outputs

The application generates:

```text
outputs/
```

containing:

* Step evolution charts
* Calories charts
* Workout frequency charts
* Regression predictions

---

## Technologies

* Python
* Pandas
* NumPy
* SciPy
* Matplotlib
* Seaborn
* Faker

---

## Future Improvements

* Machine Learning recommendation engine
* Forecasting with ARIMA/Prophet
* Interactive dashboard
* REST API
* PostgreSQL integration
* Docker support
* Automated testing

---

## Author

Fahim Coulibaly & Ibrahim Diallo

Data Analytics • Data Science • Data Engineering
