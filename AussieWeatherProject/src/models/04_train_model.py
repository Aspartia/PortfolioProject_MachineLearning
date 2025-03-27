
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.pipeline import make_pipeline
from imblearn.ensemble import BalancedRandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import matplotlib.dates as mdates


# -------------------------------------------------------------------
# Import Data
# -------------------------------------------------------------------

weather_df = pd.read_pickle("../../data/interim/03_features_dataset.pkl")

#  Humidity3pm, Sunshine, Cloud3pm , Rainfall , Cloud9am, RainToday, Pressure9am,WindGustSpeed,Pressure3pm
#WindHumidityIndex,Humidity3pm,Sunshine,HumidityRainIndex,Cloud3pm,TempRangeDiff,CloudRainIndex,Humid,TempDiff,RainCloudPressureMix

weather_df.info()
weather_df.columns
weather_df.isna().sum()

# -------------------------------------------------------------------
# Class Distribution - TargetValue
# -------------------------------------------------------------------
total = len(weather_df)
rainy = weather_df[weather_df['RainTomorrow'] == 1].shape[0]
dry = weather_df[weather_df['RainTomorrow'] == 0].shape[0]
percent_rain = 100 * rainy / total
percent_dry = 100 * dry / total

plt.figure(figsize=(6, 5))
ax = sns.countplot(x='RainTomorrow', data=weather_df, palette='coolwarm')

for p in ax.patches:
    count = p.get_height()
    percentage = 100 * count / total
    ax.annotate(f'{percentage:.2f}%',
                (p.get_x() + p.get_width() / 2., count),
                ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title("RainTomorrow Osztályeloszlás (%)", fontsize=14)
plt.xlabel("Rain Tomorrow")
plt.ylabel("Darabszám")
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# Models - Logistic Regression, DecisionTree, Random Forest Classifier, Gradient Boosting Machines (GBM), Support Vector Machines (SVM)
# -------------------------------------------------------------------

X = weather_df[['Humidity3pm', 'Sunshine', 'Cloud3pm', 'Rainfall', 'Cloud9am', 'RainToday',
                'Pressure9am', 'WindGustSpeed', 'Pressure3pm', 'WindHumidityIndex',
                'HumidityRainIndex', 'TempRangeDiff', 'CloudRainIndex', 'Humid',
                'TempDiff', 'RainCloudPressureMix']]

y = weather_df["RainTomorrow"]


# 1️⃣ SMOTE → 2️⃣ Scaler → 3️⃣ Train-Test split → 4️⃣ GridSearch

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y) 
#A fit_resample() metódust csak a SMOTE vagy RandomUnderSampler objektumon kell meghívni, és csak az adatok kiegyensúlyozása előtt, még a pipeline létrehozása előtt.

#2 UnderSampling/class weighting :(

scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)


# Gridsearch
models = {
    "Random Forest": (RandomForestClassifier(), {
        'n_estimators': [ 200],
        'max_depth': [None],
        'min_samples_leaf': [1],
        'max_features': [None]
    }), 
    "Balanced Random Forest": (BalancedRandomForestClassifier(), {
        'n_estimators': [200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2],
        'min_samples_leaf': [1, 2, 4]
    }), #{'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
    "Gradient Boosting": (GradientBoostingClassifier(), {
        'n_estimators': [ 200],
        'learning_rate': [0.1],
        'max_depth': [5]
        #'subsample': [0.8, 1.0]
    }), # {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200}
    "XGBoost": (xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss"), {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }), # {'learning_rate': 0.2, 'max_depth': 10, 'n_estimators': 200, 'subsample': 1.0}
    "LightGBM": (lgb.LGBMClassifier(), {
        'n_estimators': [100, 200],
        'max_depth': [-1, 10, 20],
        'learning_rate': [0.01, 0.1],
        'num_leaves': [31, 50],
        'subsample': [0.8, 1.0]
    }) # {'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 200, 'num_leaves': 50, 'subsample': 0.8}
} #No: MLP Neural Net, Balanced Random Forest, Easy Ensemble, Gaussian NB, Logistic Regression, Decision Tree


results = {}

for name, (model, params) in models.items():
    print(f"\n🔍 Modell tanítása: {name}")
    grid = GridSearchCV(model, param_grid=params, cv=3, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    report = classification_report(y_test, y_pred, output_dict=True)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)


    results[name] = {
        "model": best_model,
        "best_params": grid.best_params_,
        "accuracy": acc,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "fpr": fpr,
        "tpr": tpr,
        "confusion_matrix": cm,
        "report": report
    }

    print(f"✅ Legjobb paraméterek: {grid.best_params_}")
    print(f"📊 Pontosság: {acc:.4f}")
    print(f"🎯 F1-score: {f1:.4f}")
    # print(f"📈 ROC AUC score: {roc_auc:.4f}")
    # print(f"📉 ROC Curve - FPR (False Positive Rate): {fpr}")
    # print(f"📈 ROC Curve - TPR (True Positive Rate): {tpr}")
    # print(f"🎯 ROC Curve - Thresholds: {thresholds}")
    print(f"📄 Classification Report:\n{results[name]['report']}")
    print(f"🧮 Confusion Matrix:\n{cm}")


    plt.figure(figsize=(3,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Rain', 'Rain'], yticklabels=['No Rain', 'Rain'])
    plt.title(f'Confusion Matrix – {name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.show()
    
    # -------------------------------------------------------------------
    # ROC curve
    # -------------------------------------------------------------------
    if hasattr(best_model, "predict_proba"):
        y_proba = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Véletlenszerű osztályozás (AUC=0.5)')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate (Recall)")
        plt.title(f"ROC görbe – {name}")
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()   
        

# -------------------------------------------------------------------
# Visualization for Comparing the True and Predicted Data
# -------------------------------------------------------------------

def plot_moving_average_comparison(y_true, y_pred, model_name="Modell", window_size=100):
    y_true_np = np.array(y_true).astype(int)
    y_pred_np = np.array(y_pred).astype(int)
    x = np.arange(len(y_true_np))

    true_ma = pd.Series(y_true_np).rolling(window=window_size, min_periods=1).mean()
    pred_ma = pd.Series(y_pred_np).rolling(window=window_size, min_periods=1).mean()

    plt.figure(figsize=(20, 6))
    plt.plot(x, true_ma, label='Valós címkék – Csúszó átlag', color='black', linewidth=2)
    plt.plot(x, pred_ma, label=f'Predikció ({model_name}) – Csúszó átlag', color='orange', linewidth=2, alpha=0.7)

    plt.title(f"{model_name}: Csúszó átlagos összehasonlítás (ablak = {window_size})")
    plt.xlabel("Minta indexe")
    plt.ylabel("Átlagos címke érték (0–1)")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_moving_average_comparison(y_test, y_pred, model_name="XGBoost", window_size=200)



# -------------------------------------------------------------------
# RainTomorrow?
# -------------------------------------------------------------------

# => 04_predict_model




# -------------------------------------------------------------------
# Export Data
# -------------------------------------------------------------------

export_package = {
    "models": {name: grid.best_estimator_ for name, (model, _) in models.items()},
    "results": results,
    "features": X.columns.tolist()
}

joblib.dump(value=export_package, filename="../../data/processed/04_model_dataset.pkl")


