# NextGen-EV

NextGen-EV is a lightweight EV analytics system that helps riders and manufacturers estimate real-world electric vehicle range, charging time, and efficiency from practical vehicle and route inputs.

The project keeps the original static UI style, removes login friction, and adds a real machine learning backend powered by a trained Random Forest regression model.

## A. Project Overview

NextGen-EV provides two direct-access modes:

- **User Mode**: riders can enter vehicle and trip conditions to get range, charging, and efficiency insights.
- **Manufacturer Mode**: EV teams can compare design scenarios and understand how battery capacity, weight, speed, temperature, and terrain affect performance.

No authentication is required. The app is designed for fast demos, recruiter review, and practical EV analytics.

## B. Problem

Electric vehicle users often face uncertainty around:

- real-world range under different speed, temperature, and terrain conditions
- inefficient charging planning
- limited predictive insights before a ride
- unclear tradeoffs between battery size, vehicle weight, and performance

Manufacturers also need quick scenario tools to evaluate design choices without complex simulation software.

## C. Solution

NextGen-EV solves this with:

- a machine learning range prediction API
- direct User Mode and Manufacturer Mode dashboards
- lightweight static frontend pages
- practical output cards for range, charging time, and efficiency
- scenario comparison for manufacturer-style design analysis

## D. Tech Stack

**Frontend:**
- HTML5
- CSS3
- Vanilla JavaScript
- Chart.js on existing analysis pages

**Backend:**
- Python
- Flask
- Flask-CORS

**ML:**
- scikit-learn
- RandomForestRegressor
- joblib model serialization
- generated EV range dataset in CSV format

## E. Features

- Direct User Mode and Manufacturer Mode access
- EV range prediction in kilometers
- charging time estimation in hours
- efficiency estimation in percent
- practical recommendation text
- manufacturer scenario comparison
- existing model management, diagnosis, market intelligence, review analysis, and notification pages preserved
- Firebase authentication removed

## F. Innovation

- Real ML model integration instead of random or dummy calculations
- Lightweight architecture that stays easy to run locally
- Practical EV inputs: battery capacity, vehicle weight, speed, temperature, terrain, charger power, and battery state of charge
- Prediction outputs that use real units and sensible constraints

## G. Why This Project Stands Out

- Real-world EV relevance
- Clean UX with no login barrier
- Recruiter-friendly full-stack + ML architecture
- Maintains the existing visual theme while improving usability
- Simple enough to demo quickly, but realistic enough to discuss technically

## H. Flow Diagram

```text
User / Manufacturer
        |
        v
Frontend Input Form
        |
        v
Flask /predict API
        |
        v
Random Forest ML Model
        |
        v
Range + Efficiency + Charging Time
        |
        v
Result Cards + Recommendation
```

## I. Installation

1. Clone the repository.

```bash
git clone https://github.com/deepika8mandakala/NextGen-EV.git
cd NextGen-EV
```

2. Install Python dependencies.

```bash
python -m pip install -r requirements.txt
```

3. Train or regenerate the ML model.

```bash
python models/train_model.py
```

This creates:

- `data/ev_range_dataset.csv`
- `models/ev_model.pkl`

4. Start the backend API.

```bash
python backend/app.py
```

The API runs at:

```text
http://127.0.0.1:5000
```

5. Open the frontend.

Open:

```text
frontend/index.html
```

Then choose:

- User Mode
- Manufacturer Mode
- ML Range Prediction

## API Example

Endpoint:

```text
POST /predict
```

Sample request:

```json
{
  "battery_capacity_kwh": 3.7,
  "vehicle_weight_kg": 115,
  "speed_kmph": 55,
  "temperature_c": 28,
  "terrain": "mixed",
  "charger_kw": 1.0,
  "battery_soc_percent": 80
}
```

Sample response:

```json
{
  "predicted_range_km": 139.2,
  "efficiency_percent": 89.6,
  "charging_time_hours": 0.83,
  "recommendation": "Strong setup for this use case. Keep tyre pressure and charge cycles consistent to preserve the predicted range."
}
```

## J. Future Work

- real-time vehicle telemetry
- IoT sensor integration
- battery health prediction
- route-aware range prediction
- live weather integration
- deployment of backend API to a cloud service

## K. Demo Section

Add screenshots here:

```text
static/demo/home.png
static/demo/user-mode.png
static/demo/manufacturer-mode.png
static/demo/prediction-results.png
```

## Updated Structure

```text
NextGen-EV/
  backend/
    app.py
  data/
    ev_range_dataset.csv
  frontend/
    index.html
    user-dashboard.html
    dashboard.html
    model.html
    vehicle-diagnosis.html
    model-management.html
    review-analysis.html
    market-intelligence.html
    smart-alerts.html
    ...
  models/
    train_model.py
    ev_model.pkl
  static/
    images/
      biker.png
      photo.jpg
      ...
  README.md
  requirements.txt
```

## Removed Authentication Files

- `frontend/user-login.html`
- `frontend/user-login1.html`

Firebase SDK usage and login/register flows were removed to simplify UX and avoid exposing auth configuration in a static frontend.
