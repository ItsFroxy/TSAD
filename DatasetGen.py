import numpy as np
import pandas as pd
import json
from datetime import timedelta

np.random.seed(42)

# Configuration
num_patients = 50
minutes_per_day = 24 * 60
days = 7
total_minutes = minutes_per_day * days

# Define activity levels
activity_levels = ["Idle", "Light", "Moderate", "Vigorous"]
activity_probs = np.array([0.5, 0.2, 0.2, 0.1])

# Define patient profiles
patients = []

for pid in range(1, num_patients + 1):
    age = np.random.randint(18, 80)
    gender = np.random.choice(["M", "F"])
    height_cm = np.round(np.random.normal(175 if gender == "M" else 162, 7), 1)
    weight_kg = np.round(np.random.normal(70 if gender == "M" else 60, 10), 1)
    patient = {
        "patient_id": f"P{pid:03}",
        "gender": gender,
        "age": age,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "blood_type": np.random.choice(["O+", "A+", "B+", "AB+", "O-", "A-", "B-", "AB-"]),
        "allergies": [],
        "chronic_conditions": [],
        "medications": [],
        "time_series": []
    }

    # Anomaly injection flags
    inject_anomaly = np.random.rand() < 0.3  # 30% of patients will have hidden anomalies
    anomaly_start = np.random.randint(1 * 1440, 5 * 1440) if inject_anomaly else -1  # day 2–5
    anomaly_duration = np.random.randint(30, 90)  # 30–90 minutes

    # Generate time-series
    start_time = pd.Timestamp("2025-01-01")
    for minute in range(total_minutes):
        timestamp = start_time + timedelta(minutes=minute)
        hour = timestamp.hour

        sleeping = hour >= 23 or hour < 6
        activity = "Idle" if sleeping else np.random.choice(activity_levels, p=activity_probs)
        in_anomaly = inject_anomaly and anomaly_start <= minute < (anomaly_start + anomaly_duration)

        # Base values
        heart_rate = np.random.normal(60, 5)
        spo2 = np.random.normal(98, 1)
        respiratory_rate = np.random.normal(14, 2)
        systolic_bp = np.random.normal(115, 5)
        diastolic_bp = np.random.normal(75, 3)
        temperature = np.random.normal(36.5, 0.2)
        skin_temp = temperature - 0.5
        hrv = np.random.normal(70, 10)
        eda = np.random.normal(0.1, 0.05)
        steps = 0
        calories_burned = 0.0
        sleep_stage = np.random.choice(["Light", "Deep", "REM"]) if sleeping else None
        state_context = "deep_sleep" if sleeping else ("exercise" if activity in ["Moderate", "Vigorous"] else "resting")

        # Adjust by activity
        if activity == "Light":
            heart_rate += 15
            steps = np.random.poisson(60)
            calories_burned = round(steps * 0.03, 2)
        elif activity == "Moderate":
            heart_rate += 30
            steps = np.random.poisson(100)
            calories_burned = round(steps * 0.04, 2)
            respiratory_rate += 5
            eda += 0.3
        elif activity == "Vigorous":
            heart_rate += 50
            steps = np.random.poisson(140)
            calories_burned = round(steps * 0.05, 2)
            respiratory_rate += 10
            eda += 0.5
            temperature += 0.3

        # Inject medically realistic hidden anomalies
        if in_anomaly:
            # Drop in oxygen saturation → higher HR, RR, EDA
            spo2 -= np.random.uniform(4, 7)  # drop by 4–7%
            heart_rate += np.random.uniform(15, 30)
            respiratory_rate += np.random.uniform(5, 10)
            eda += np.random.uniform(0.5, 1.0)
            hrv -= np.random.uniform(15, 25)
            state_context = "anomaly"

        # Clamp and round
        spo2 = int(np.clip(spo2, 85, 100))
        heart_rate = int(np.clip(heart_rate, 40, 200))
        respiratory_rate = int(np.clip(respiratory_rate, 8, 50))
        systolic_bp = int(np.clip(systolic_bp, 90, 180))
        diastolic_bp = int(np.clip(diastolic_bp, 50, 120))
        hrv = int(np.clip(hrv, 10, 150))
        eda = float(np.round(np.clip(eda, 0.01, 5.0), 2))
        temperature = float(np.round(temperature, 1))
        skin_temp = float(np.round(skin_temp, 1))

        patient["time_series"].append({
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
            "heart_rate": heart_rate,
            "spo2": spo2,
            "respiratory_rate": respiratory_rate,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "temperature": temperature,
            "hrv": hrv,
            "steps": steps,
            "calories_burned": calories_burned,
            "skin_temp": skin_temp,
            "eda": eda,
            "sleep_stage": sleep_stage,
            "state_context": state_context
        })

    patients.append(patient)

# Save to JSON
with open("synthetic_realistic_unlabeled_dataset.json", "w") as f:
    json.dump(patients, f, indent=2)

print("✅ Dataset saved as 'synthetic_realistic_unlabeled_dataset.json'")
