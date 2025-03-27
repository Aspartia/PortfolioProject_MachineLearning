import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

# Betöltés
weather_df = pd.read_pickle("../../data/interim/03_features_dataset.pkl")

# Feature + TargetValue
X = weather_df[['Humidity3pm', 'Sunshine', 'Cloud3pm', 'Rainfall', 'Cloud9am', 'RainToday',
                'Pressure9am', 'WindGustSpeed', 'Pressure3pm', 'WindHumidityIndex',
                'HumidityRainIndex', 'TempRangeDiff', 'CloudRainIndex', 'Humid',
                'TempDiff', 'RainCloudPressureMix']]
y = weather_df["RainTomorrow"]

# SMOTE + Scaler
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Gradient Boosting Train
gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5)
gb_model.fit(X_scaled, y_resampled)

# Kiválasztott városok és dátumintervallum
date_rng = pd.date_range(start='2017-01-01', end='2017-06-30', freq='D')
cities = weather_df['Location'].unique()



# Szűrés az eredeti df-ből csak a kijelölt városokra
city_df_all = weather_df[weather_df["Location"].isin(cities)].copy()
city_df_all["Date"] = pd.to_datetime(city_df_all["Date"])
city_df_all = city_df_all[city_df_all["Date"].isin(date_rng)].copy()
city_df_all = city_df_all.sort_values(["Location", "Date"]).reset_index(drop=True)

# Hiányzó értékek eltávolítása és scaling
city_df_features = city_df_all[['Humidity3pm', 'Sunshine', 'Cloud3pm', 'Rainfall', 'Cloud9am', 'RainToday',
                                'Pressure9am', 'WindGustSpeed', 'Pressure3pm', 'WindHumidityIndex',
                                'HumidityRainIndex', 'TempRangeDiff', 'CloudRainIndex', 'Humid',
                                'TempDiff', 'RainCloudPressureMix']].copy()
city_df_features = scaler.transform(city_df_features)
city_df_all["Predicted"] = gb_model.predict(city_df_features)
city_df_all["lower"] = (city_df_all["Predicted"] - 0.2).clip(0, 1)
city_df_all["upper"] = (city_df_all["Predicted"] + 0.2).clip(0, 1)

# Vizu
horizon = 7
for city in cities:
    df_city = city_df_all[city_df_all["Location"] == city].copy()
    tail_df = df_city.tail(horizon * 7)
    dates = mdates.date2num(tail_df["Date"])

    plt.figure(figsize=(14, 6))
    plt.plot(tail_df["Date"], tail_df["RainTomorrow"], label="Actual", color="black")
    plt.plot(tail_df["Date"], tail_df["Predicted"], label="Predicted", color="dodgerblue")
    plt.fill_between(dates, tail_df["lower"], tail_df["upper"], color='blue', alpha=0.1, label="Confidence interval")
    plt.title(f"{city} – Eső előrejelzés ({horizon} nap)")
    plt.xlabel("Dátum")
    plt.ylabel("RainTomorrow")
    plt.ylim(-0.1, 1.1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    #plt.savefig(f"../../reports/figures/will_it_rain/rain_{city}.png")
    plt.show()
