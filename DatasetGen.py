import numpy as np
import pandas as pd
import json
from datetime import timedelta

np.random.seed(42)

# Configuration
NUM_PATIENTS = 30
DAYS = 7
MINUTES_PER_DAY = 24 * 60
RECORDS_PER_PATIENT = DAYS * MINUTES_PER_DAY

activity_levels = ["Idle", "Light", "Moderate", "Vigorous"]
base_activity_probs = np.array([0.5, 0.2, 0.2, 0.1])
possible_allergies = ["Penicillin", "Peanuts", "Shellfish", "Aspirin", "Pollen", "Dust"]
blood_types = ["O+", "A+", "B+", "AB+", "O-", "A-", "B-", "AB-"]
blood_type_probs = np.array([0.374, 0.357, 0.085, 0.034, 0.066, 0.063, 0.015, 0.006])
blood_type_probs /= blood_type_probs.sum()

def assign_condition(age):
    if age < 40:
        return np.random.choice(["None", "Hypertension", "Diabetes"], p=[0.90, 0.05, 0.05])
    elif age < 60:
        return np.random.choice(["None", "Hypertension", "Diabetes", "Hypertension+Diabetes"], p=[0.60, 0.20, 0.15, 0.05])
    else:
        return np.random.choice(["None", "Hypertension", "Diabetes", "Hypertension+Diabetes"], p=[0.40, 0.25, 0.25, 0.10])

def generate_patient(person_id, inject_anomaly=False):
    age = np.random.randint(18, 80)
    gender = np.random.choice(["M", "F"])
    condition = assign_condition(age)
    
    height_cm = np.random.normal(175 if gender=="M" else 162, 10 if gender=="M" else 8)
    height_cm = float(np.clip(height_cm, 140, 200))

    bmi = 24 if condition == "None" else (28 if condition in ["Hypertension", "Diabetes"] else 30)
    bmi += np.random.normal(0, 3)
    bmi = np.clip(bmi, 18, 40)
    weight_kg = round(bmi * ((height_cm/100)**2), 1)

    blood_type = np.random.choice(blood_types, p=blood_type_probs)
    allergies = []
    if np.random.rand() < 0.2:
        allergies = list(np.random.choice(possible_allergies, size=np.random.randint(1,3), replace=False))

    chronic_conditions = [] if condition == "None" else condition.split("+")
    medications = []
    if "Hypertension" in chronic_conditions:
        medications.append("Metoprolol")
    if "Diabetes" in chronic_conditions:
        medications.append("Metformin")

    resting_hr = int(np.clip(60 + 0.1*age + (3 if gender=="F" else 0) + (5 if condition!="None" else 0), 50, 100))
    max_hr = max(int(np.clip(220 - age + np.random.randint(-5,6) - (5 if "Hypertension" in chronic_conditions else 0), 100, 220)), resting_hr + 20)

    patient_data = {
        "patient_id": f"P{person_id:03d}",
        "gender": gender,
        "age": age,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "blood_type": blood_type,
        "allergies": allergies,
        "chronic_conditions": chronic_conditions,
        "medications": medications,
        "time_series": []
    }

    start_time = pd.Timestamp("2025-01-01")
    for i in range(RECORDS_PER_PATIENT):
        t = start_time + timedelta(minutes=i)
        hour = t.hour
        sleeping = hour >= 23 or hour < 6
        activity = "Idle" if sleeping else np.random.choice(activity_levels, p=base_activity_probs)
        
        # Vital signs
        heart_rate = resting_hr + (5 if activity == "Light" else 15 if activity == "Moderate" else 25 if activity == "Vigorous" else 0) + np.random.normal(0, 2)
        spo2 = 98 - (2 if activity in ["Moderate","Vigorous"] else 0) + np.random.normal(0,1)
        respiratory_rate = 14 + (5 if activity == "Light" else 10 if activity == "Moderate" else 20 if activity == "Vigorous" else 0) + np.random.normal(0,1)
        hrv = 80 - (30 if activity in ["Moderate","Vigorous"] else 0) + np.random.normal(0,5)
        eda = 0.1 + (0.5 if activity == "Vigorous" else 0.2) + np.random.normal(0, 0.05)
        temperature = 36.5 + (0.5 if activity in ["Moderate", "Vigorous"] else 0) + np.random.normal(0, 0.1)
        skin_temp = temperature - (1.5 if activity == "Vigorous" else 0.5)
        steps = np.random.poisson(1 if activity == "Idle" else 60)
        calories = steps * (0.04 if activity in ["Moderate", "Vigorous"] else 0.02)

        state = "anomaly" if inject_anomaly and 2000 < i < 2020 else (
            "exercise" if activity in ["Moderate", "Vigorous"] else
            "light_activity" if activity == "Light" else
            "resting"
        )

        # Inject realistic anomaly (e.g., low SpO2 → correlated vitals)
        if inject_anomaly and 2000 < i < 2020:
            spo2 = 89
            heart_rate += 20
            respiratory_rate += 10
            eda += 1.0
            hrv -= 30
            state = "anomaly"

        patient_data["time_series"].append({
            "timestamp": str(t),
            "heart_rate": round(heart_rate),
            "spo2": int(np.clip(spo2, 85, 100)),
            "respiratory_rate": int(np.clip(respiratory_rate, 10, 40)),
            "systolic_bp": int(np.random.normal(120, 10)),
            "diastolic_bp": int(np.random.normal(80, 5)),
            "temperature": round(np.clip(temperature, 35.5, 38.5), 1),
            "hrv": int(np.clip(hrv, 5, 150)),
            "steps": int(steps),
            "calories_burned": round(calories, 2),
            "skin_temp": round(skin_temp, 1),
            "eda": round(eda, 2),
            "state_context": state
        })

    return patient_data

# --- Generate Patients ---
healthy_patients = [generate_patient(i+1, inject_anomaly=False) for i in range(20)]
mixed_patients = healthy_patients + [generate_patient(i+21, inject_anomaly=True) for i in range(10)]

# --- Save JSON ---
with open("healthy_data.json", "w") as f:
    json.dump(healthy_patients, f, indent=2)

with open("mixed_data.json", "w") as f:
    json.dump(mixed_patients, f, indent=2)

print("✅ Saved healthy_data.json (20 patients) and mixed_data.json (30 patients total)")
