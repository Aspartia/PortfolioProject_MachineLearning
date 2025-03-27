import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# -------------------------------------------------------------------
# Import Data
# -------------------------------------------------------------------

weather_df= pd.read_pickle("../../data/interim/02_outliers_dataset_KNN_df_IQR.pkl")
KNN_df_chauv= pd.read_pickle("../../data/interim/02_outliers_dataset_KNN_df_chauv.pkl")
KNN_df_Zscore= pd.read_pickle("../../data/interim/02_outliers_dataset_KNN_df_Zscore.pkl")

imputer_df_IQR= pd.read_pickle("../../data/interim/02_outliers_dataset_imputer_df_IQR.pkl")
imputer_df_chauv= pd.read_pickle("../../data/interim/02_outliers_dataset_imputer_df_chauv.pkl")
imputer_df_Zscore= pd.read_pickle("../../data/interim/02_outliers_dataset_imputer_df_Zscore.pkl")

function_df_IQR= pd.read_pickle("../../data/interim/02_outliers_dataset_function_df_IQR.pkl")
function_df_chauv= pd.read_pickle("../../data/interim/02_outliers_dataset_function_df_chauv.pkl")
function_df_Zscore= pd.read_pickle("../../data/interim/02_outliers_dataset_function_df_Zscore.pkl")

# -------------------------------------------------------------------
# Info
# -------------------------------------------------------------------
weather_df.info()
weather_df.isna().sum()
weather_df.columns


# -------------------------------------------------------------------
# Feature Building (Calculating and Transforming Numeric Features) *make_datasetbe
# -------------------------------------------------------------------

def create_weather_features(weather_df):

    df = weather_df.copy()

    # **1. Időalapú változók**
    df['Month'] = df['Date'].dt.month
    df['Season'] = (df['Month'] % 12 // 3) # 0: 'Winter', 1: 'Spring', 2: 'Summer', 3: 'Autumn'
    #df['Season_Name'] = df['Month'].map(lambda x: {0: 'Winter', 1: 'Spring', 2: 'Summer', 3: 'Autumn'}[(x % 12) // 3])
    df['Day_of_Year'] = df['Date'].dt.dayofyear

    # **2. Szélirány numerikus átalakítása (megtartva az eredeti oszlopokat)**
    wind_directions = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
    for col in wind_directions:
        df[col + '_sin'] = np.sin(np.radians(df[col]))
        df[col + '_cos'] = np.cos(np.radians(df[col]))

    # **3. Időbeli különbségek kiszámítása**
    # def safe_divide(numerator, denominator):
    #     """Biztonságos osztás, elkerüli a 0-val való osztást és a NaN hibákat."""
    #     return np.where((denominator == 0) | (pd.isna(denominator)), 0, numerator / (denominator + 1))

    # df['TempRangeRatio'] = safe_divide(df['MaxTemp'], df['MinTemp'])
    # df['TempChangeRatio'] = safe_divide(df['Temp3pm'], df['Temp9am'])
    # df['PressureRatio'] = safe_divide(df['Pressure3pm'], df['Pressure9am'])
    # df["HumidityRatio"] = safe_divide(df["Humidity3pm"], df["Humidity9am"])
    # df['WindSpeedRatio'] = safe_divide(df['WindSpeed3pm'], df['WindSpeed9am'])
    # df['CloudRatio'] = safe_divide(df['Cloud3pm'], df['Cloud9am'])

    df['TempRangeDiff'] = df['MaxTemp'] - df['MinTemp']
    df['TempDiff'] = df['Temp3pm'] - df['Temp9am']
    df['PressureDiff'] = df['Pressure3pm'] - df['Pressure9am']
    df['WindSpeedDiff'] = df['WindSpeed3pm'] - df['WindSpeed9am']
    df['CloudDiff'] = df['Cloud3pm'] - df['Cloud9am']


    # **4. Időbeli aggregációk (Moving Averages)**
    df['Rainfall_last_7days'] = df['Rainfall'].rolling(window=7, min_periods=1).mean()
    df['Rainfall_last_30days'] = df['Rainfall'].rolling(window=30, min_periods=1).mean()

    # **5. Interakciós változók** Ezt még finomítani!
    df['Rainy'] = (df['Temp3pm'] - df['Temp9am']) * df['Rainfall']
    df['Storm'] = df['WindGustSpeed'] * df['Rainfall']
    df['Humid'] = df['Humidity3pm'] * df['Temp3pm']
    df['HumidityRainIndex'] = df['Humidity3pm'] * df['Rainfall']
    df['Thunderstorm'] = (df['Humidity3pm'] - df['Humidity9am']) * df['Temp3pm'] 
    df['PressureTempIndex'] = df['Pressure3pm'] * df['Temp3pm']
    df['WindPressureChange'] = (df['Pressure3pm'] - df['Pressure9am']) * df['WindGustSpeed']
    df['PressureHumidityDiff'] = (df['Pressure3pm'] - df['Pressure9am']) * (df['Humidity3pm'] - df['Humidity9am'])
    df['CloudPressureChange'] =  (df['Pressure9am'] - df['Pressure3pm'])* (df['Cloud3pm'] - df['Cloud9am'])
    df['HumidityTempDiffIndex'] = (df['Humidity3pm'] - df['Humidity9am']) * (df['Temp3pm'] - df['Temp9am'])
    df['TempWindIndex'] = (df['Temp3pm'] - df['Temp9am']) * df['WindGustSpeed']
    df['CloudTempIndex'] = df['Cloud3pm'] * df['Temp3pm']
    df['WindSpeed'] = df['WindGustDir_sin'] * df['WindGustSpeed']
    df['WindHumidityIndex'] = df['WindGustSpeed'] * df['Humidity3pm']
    df['CloudRainIndex'] = df['Cloud3pm'] * df['Rainfall']
    df['RainCloudPressureMix'] = df['Rainfall'] * df['Cloud3pm'] * (df['Pressure9am'] - df['Pressure3pm'])
    df['RainWindCombinedEffect'] = df['Rainfall'] * df['WindSpeed3pm'] * df['Cloud3pm']

    # **6. Logaritmikus és négyzetgyök transzformációk**
    df['Log_Rainfall'] = np.log1p(df['Rainfall'])  # log(1 + x) az elkerülésére
    df['Log_Evaporation'] = np.log1p(df['Evaporation']) 
    df['Sqrt_WindGustSpeed'] = np.log1p(df['WindGustSpeed'])
    df['Sqrt_Humidity9am'] = np.sqrt(df['Humidity9am'])

    return df


weather_df_features = create_weather_features(weather_df)


# -------------------------------------------------------------------
# Normal Distribution - Mely oszlopok fontosak a modellépítés szempontjából?
# -------------------------------------------------------------------
#Ha az adatokat külön szedjük (RainTomorrow = 0 és RainTomorrow = 1),
# akkor csak azt látjuk, hogy az egyes bemeneti változók hogyan viselkednek az esős és száraz napokon.
#Megnézzük, hogy a numerikus oszlopok különbözően oszlanak-e el esős és száraz napokon.
#Ha jelentős különbség van a két eloszlás között, akkor a "WindGustSpeed" jó prediktor lehet az eső előrejelzésére.
#A, t-próba (Student's t-test): Megmutatja, hogy a két csoport átlaga közötti különbség statisztikailag szignifikáns-e.
    #Ha p-érték nagyon kicsi, ami azt jelenti, hogy rendkívül erős statisztikai bizonyíték van arra, hogy a két csoport különbözik.
    #Ha a p-érték < 0.05, akkor az esős és száraz napok széllökési eloszlása valóban különböző.
#B, Kolmogorov–Smirnov teszt: Az eloszlások különbségeit méri.
# A KS-érték 0 és 1 között van, ahol:
    #Minél közelebb van 0-hoz, annál kevésbé különböznek az eloszlások.
    #Minél közelebb van 1-hez, annál nagyobb a különbség

from analyzing_correlation import compare_numeric_features_with_target, analyze_multicollinearity
from analyzing_correlation import analyze_linearity_correlation, analyze_linearity_correlation2

compare_numeric_features_with_target(weather_df_features)

# Ezek kellenek: Rainfall, Sunshine, WindGustSpeed, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm
# + TempRangeDiff, TempDiff, HumidityDiff, Rainfall_last_7days, Rainfall_last_30days, HumidityTempIndex, WindRainIndex, PressureTempIndex, LogRainfall, SqrtWindSpeed

# -------------------------------------------------------------------
# Multikollinearitás - Only Between Input Labels 
# -------------------------------------------------------------------

analyze_multicollinearity(weather_df_features)
# Ezek kellenek: MinTemp, MaxTemp, Rainfall, Sunshine, WindGustSpeed, Humidity9am, Humidity3pm, Pressure3pm, Cloud9am, Cloud3pm

# -------------------------------------------------------------------
# Linearity & Correlation (Target and Input Labels)
# -------------------------------------------------------------------

correlation_results  = analyze_linearity_correlation(weather_df_features)
correlation_results  = analyze_linearity_correlation2(weather_df_features)

# No correlation between the targetvalue and the input labels. 

# -------------------------------------------------------------------
# Feature Importance (SelectKBest, Random Forest Feature Importance) - On Original Features
# -------------------------------------------------------------------

def feature_selection_analysis_original(df, target_col="RainTomorrow", k=20, top_n=10):
    X = df.drop(columns=[target_col, "Date", "Location"]) 
    y = df[target_col]

    mi_selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_mi_selected = mi_selector.fit_transform(X, y)

    selected_features_mi = X.columns[mi_selector.get_support()]
    feature_scores_mi = mi_selector.scores_[mi_selector.get_support()]

    mi_df = pd.DataFrame({"Feature": selected_features_mi, "Importance": feature_scores_mi})
    mi_df = mi_df.sort_values(by="Importance", ascending=False)

    print("Kiválasztott változók (Mutual Information):", mi_df.head(top_n).to_string(index=False))

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    rf_df = pd.DataFrame({"Feature": X.columns, "Importance": rf_model.feature_importances_})
    rf_df = rf_df.sort_values(by="Importance", ascending=False)

    print("Kiválasztott változók (Random Forest):", rf_df.head(top_n).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.barplot(x="Importance", y="Feature", data=mi_df, palette="Blues", ax=axes[0], legend=False, hue='Feature')
    axes[0].set_title("Mutual Information Feature Importance")
    axes[0].set_xlabel("Fontosság")
    axes[0].set_ylabel("Változó")

    sns.barplot(x="Importance", y="Feature", data=rf_df[:k], palette="Reds", ax=axes[1], legend=False, hue='Feature')
    axes[1].set_title("Random Forest Feature Importance")
    axes[1].set_xlabel("Fontosság")
    axes[1].set_ylabel("Változó")

    plt.tight_layout()
    plt.show()

    return {
        "mutual_info_features": list(mi_df["Feature"].head(top_n)),
        "random_forest_features": list(rf_df["Feature"].head(top_n)),
        "mi_importance_df": mi_df,
        "rf_importance_df": rf_df
    }


feature_selection_original = feature_selection_analysis_original(weather_df)

#(Mutual Information): Humidity3pm, Sunshine, Cloud3pm , Rainfall , Cloud9am, RainToday, Humidity9am, Pressure9am, WindGustSpeed, Pressure3pm  
#(Random Forest):Humidity3pm, Sunshine, Pressure3pm, Cloud3pm, Pressure9am, WindGustSpeed, Humidity9am, Rainfall, Temp3pm, MinTemp



# -------------------------------------------------------------------
# Feature Importance (SelectKBest, Random Forest Feature Importance) - On New Features 
# -------------------------------------------------------------------

def feature_selection_analysis_new(df, target_col="RainTomorrow", k=20, top_n=10):
    df = weather_df_features.copy()
    print("\n Feature Selection elemzés elindítva...\n")

    drop_cols = [target_col]
    for col in ["Date", "Location"]:
        if col in df.columns:
            drop_cols.append(col)

    X = df.drop(columns=drop_cols)
    y = df[target_col]

    if y.nunique() > 2:
        print("Figyelmeztetés: A célváltozó több kategóriát tartalmaz! Automatikusan binárisba alakítom.")
        le = LabelEncoder()
        y = le.fit_transform(y)

    max_features = min(k, X.shape[1])
    if k > X.shape[1]:
        print(f"Figyelmeztetés: Az elérhető változók száma ({X.shape[1]}) kisebb, mint a kívánt k érték ({k}). Az új érték: {max_features}")


    mi_selector = SelectKBest(score_func=mutual_info_classif, k=max_features)
    X_mi_selected = mi_selector.fit_transform(X, y)

    selected_features_mi = X.columns[mi_selector.get_support()]
    feature_scores_mi = mi_selector.scores_[mi_selector.get_support()]

    mi_df = pd.DataFrame({"Feature": selected_features_mi, "Importance": feature_scores_mi})
    mi_df = mi_df.sort_values(by="Importance", ascending=False)
    print("\n**Top 10 változó (Mutual Information):**")
    print(mi_df.head(top_n).to_string(index=False))

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    rf_df = pd.DataFrame({"Feature": X.columns, "Importance": rf_model.feature_importances_})
    rf_df = rf_df.sort_values(by="Importance", ascending=False).head(max_features)
    print("\n**Top 10 változó (Random Forest):**")
    print(rf_df.head(top_n).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.barplot(x="Importance", y="Feature", data=mi_df.head(top_n), palette="Blues", ax=axes[0], legend=False, hue='Feature')
    axes[0].set_title("Mutual Information Feature Importance (Top 10)")
    axes[0].set_xlabel("Fontosság")
    axes[0].set_ylabel("Változó")

    sns.barplot(x="Importance", y="Feature", data=rf_df.head(top_n), palette="Reds", ax=axes[1], legend=False, hue='Feature')
    axes[1].set_title("Random Forest Feature Importance (Top 10)")
    axes[1].set_xlabel("Fontosság")
    axes[1].set_ylabel("Változó")

    plt.tight_layout()
    plt.show()

    return {
        "mutual_info_features": list(mi_df["Feature"].head(top_n)),
        "random_forest_features": list(rf_df["Feature"].head(top_n)),
        "mi_importance_df": mi_df,
        "rf_importance_df": rf_df
    }

feature_selection_new = feature_selection_analysis_new(weather_df_features)

#WindHumidityIndex,Humidity3pm,Sunshine,HumidityRainIndex,Cloud3pm,TempRangeDiff,CloudRainIndex,Humid,TempDiff,RainCloudPressureMix

# -------------------------------------------------------------------
# Final Features
# -------------------------------------------------------------------

#  Humidity3pm, Sunshine, Cloud3pm , Rainfall , Cloud9am, RainToday, Pressure9am,WindGustSpeed,Pressure3pm
#WindHumidityIndex,Humidity3pm,Sunshine,HumidityRainIndex,Cloud3pm,TempRangeDiff,CloudRainIndex,Humid,TempDiff,RainCloudPressureMix


# -------------------------------------------------------------------
#  Export Data
# -------------------------------------------------------------------
weather_df_features.to_pickle("../../data/interim/03_features_dataset.pkl") 


