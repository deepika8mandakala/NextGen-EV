from pathlib import Path
import csv
import random
import site
import sys

user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.append(user_site)

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "ev_range_dataset.csv"
MODEL_PATH = ROOT_DIR / "models" / "ev_model.pkl"

FEATURES = [
    "battery_capacity_kwh",
    "vehicle_weight_kg",
    "speed_kmph",
    "temperature_c",
    "terrain_city",
    "terrain_highway",
    "terrain_hilly",
    "terrain_mixed",
]

TERRAINS = {
    "city": 0.88,
    "highway": 0.92,
    "hilly": 0.74,
    "mixed": 1.0,
}


def realistic_range(battery_kwh, weight_kg, speed_kmph, temp_c, terrain):
    base_efficiency_km_per_kwh = 43.0

    weight_factor = max(0.62, 1 - ((weight_kg - 110) * 0.00022))
    speed_factor = max(0.68, 1 - abs(speed_kmph - 45) * 0.0045)
    temperature_factor = max(0.72, 1 - abs(temp_c - 25) * 0.006)
    terrain_factor = TERRAINS[terrain]

    noise = random.uniform(0.94, 1.06)
    predicted = (
        battery_kwh
        * base_efficiency_km_per_kwh
        * weight_factor
        * speed_factor
        * temperature_factor
        * terrain_factor
        * noise
    )
    return round(max(8, predicted), 2)


def generate_dataset(rows=900):
    random.seed(42)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    with DATA_PATH.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "battery_capacity_kwh",
            "vehicle_weight_kg",
            "speed_kmph",
            "temperature_c",
            "terrain",
            "range_km",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for _ in range(rows):
            battery = round(random.uniform(2.0, 8.0), 2)
            weight = round(random.uniform(85, 180), 1)
            speed = round(random.uniform(25, 95), 1)
            temp = round(random.uniform(5, 42), 1)
            terrain = random.choice(list(TERRAINS))
            writer.writerow(
                {
                    "battery_capacity_kwh": battery,
                    "vehicle_weight_kg": weight,
                    "speed_kmph": speed,
                    "temperature_c": temp,
                    "terrain": terrain,
                    "range_km": realistic_range(battery, weight, speed, temp, terrain),
                }
            )

        for _ in range(rows // 4):
            battery = round(random.uniform(25, 90), 2)
            weight = round(random.uniform(950, 2400), 1)
            speed = round(random.uniform(35, 120), 1)
            temp = round(random.uniform(-5, 45), 1)
            terrain = random.choice(list(TERRAINS))
            writer.writerow(
                {
                    "battery_capacity_kwh": battery,
                    "vehicle_weight_kg": weight,
                    "speed_kmph": speed,
                    "temperature_c": temp,
                    "terrain": terrain,
                    "range_km": realistic_range(battery, weight, speed, temp, terrain),
                }
            )


def load_dataset():
    if not DATA_PATH.exists():
        generate_dataset()

    rows = []
    targets = []
    with DATA_PATH.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            terrain = row["terrain"]
            features = {
                "battery_capacity_kwh": float(row["battery_capacity_kwh"]),
                "vehicle_weight_kg": float(row["vehicle_weight_kg"]),
                "speed_kmph": float(row["speed_kmph"]),
                "temperature_c": float(row["temperature_c"]),
                "terrain_city": 1.0 if terrain == "city" else 0.0,
                "terrain_highway": 1.0 if terrain == "highway" else 0.0,
                "terrain_hilly": 1.0 if terrain == "hilly" else 0.0,
                "terrain_mixed": 1.0 if terrain == "mixed" else 0.0,
            }
            rows.append([features[name] for name in FEATURES])
            targets.append(float(row["range_km"]))
    return rows, targets


def train():
    generate_dataset()
    x, y = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=160,
        max_depth=14,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Saved model: {MODEL_PATH}")
    print(f"Dataset: {DATA_PATH}")
    print(f"MAE: {mae:.2f} km")
    print(f"R2: {r2:.3f}")


if __name__ == "__main__":
    train()
