from pathlib import Path
import site
import sys

user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.append(user_site)

import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS


ROOT_DIR = Path(__file__).resolve().parents[1]
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

TERRAIN_VALUES = {"city", "highway", "hilly", "mixed"}

app = Flask(__name__)
CORS(app)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python models/train_model.py"
        )
    return joblib.load(MODEL_PATH)


model = load_model()


def as_float(payload, field, minimum, maximum):
    value = payload.get(field)
    if value is None:
        raise ValueError(f"{field} is required")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}")
    return parsed


def build_features(payload):
    terrain = str(payload.get("terrain", "mixed")).lower()
    if terrain not in TERRAIN_VALUES:
        raise ValueError("terrain must be one of city, highway, hilly, mixed")

    values = {
        "battery_capacity_kwh": as_float(payload, "battery_capacity_kwh", 1, 120),
        "vehicle_weight_kg": as_float(payload, "vehicle_weight_kg", 60, 2600),
        "speed_kmph": as_float(payload, "speed_kmph", 15, 130),
        "temperature_c": as_float(payload, "temperature_c", -10, 50),
        "terrain_city": 1.0 if terrain == "city" else 0.0,
        "terrain_highway": 1.0 if terrain == "highway" else 0.0,
        "terrain_hilly": 1.0 if terrain == "hilly" else 0.0,
        "terrain_mixed": 1.0 if terrain == "mixed" else 0.0,
    }
    return [values[name] for name in FEATURES], terrain


def estimate_efficiency(range_km, battery_capacity_kwh, terrain):
    ideal_range = battery_capacity_kwh * 42
    terrain_adjustment = {
        "city": 0.92,
        "highway": 0.96,
        "mixed": 1.0,
        "hilly": 0.82,
    }[terrain]
    efficiency = (range_km / max(ideal_range * terrain_adjustment, 1)) * 100
    return round(max(45, min(efficiency, 98)), 1)


def estimate_charging_time(battery_capacity_kwh, soc_percent, charger_kw):
    usable_gap = max(0, 100 - soc_percent) / 100
    energy_needed = battery_capacity_kwh * usable_gap
    charging_loss_factor = 1.12
    return round((energy_needed * charging_loss_factor) / max(charger_kw, 0.2), 2)


def recommendation(range_km, efficiency_percent, terrain, speed_kmph):
    if efficiency_percent >= 88 and range_km >= 100:
        return "Strong setup for this use case. Keep tyre pressure and charge cycles consistent to preserve the predicted range."
    if terrain == "hilly":
        return "Hilly terrain is reducing usable range. Consider a larger battery pack or lower average speed for this route."
    if speed_kmph > 75:
        return "High average speed is the main efficiency drag. Reducing speed by 8-10 km/h should improve range noticeably."
    if efficiency_percent < 70:
        return "Efficiency is below target. Reduce vehicle weight, improve aerodynamics, or use eco riding mode."
    return "Predicted performance is practical for daily use. Charge planning should be based on the displayed range with a 10% reserve."


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": str(MODEL_PATH)})


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(force=True) or {}
        feature_values, terrain = build_features(payload)
        battery_capacity = float(payload["battery_capacity_kwh"])
        speed = float(payload["speed_kmph"])
        charger_kw = as_float(payload, "charger_kw", 0.5, 80)
        soc_percent = as_float(payload, "battery_soc_percent", 5, 100)

        predicted_range = float(model.predict([feature_values])[0])
        predicted_range = round(max(8, min(predicted_range, battery_capacity * 58)), 1)
        efficiency = estimate_efficiency(predicted_range, battery_capacity, terrain)
        charging_time = estimate_charging_time(battery_capacity, soc_percent, charger_kw)

        return jsonify(
            {
                "predicted_range_km": predicted_range,
                "efficiency_percent": efficiency,
                "charging_time_hours": charging_time,
                "recommendation": recommendation(predicted_range, efficiency, terrain, speed),
                "units": {
                    "range": "km",
                    "charging_time": "hours",
                    "efficiency": "percent",
                },
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Prediction service error: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
