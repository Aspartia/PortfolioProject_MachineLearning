       
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# -------------------------------------------------------------------
# KNN Imputer
# -------------------------------------------------------------------

def knn_impute(weather_df, n_neighbors=5):
    label_encoders = {}
    
    categorical_cols = weather_df.select_dtypes(include=['object']).columns

    if len(categorical_cols) > 0:
        for col in categorical_cols:
            le = LabelEncoder()
            weather_df[col] = le.fit_transform(weather_df[col].astype(str))
            label_encoders[col] = le  
        
        weather_df[categorical_cols] = weather_df[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))

    numeric_cols = weather_df.select_dtypes(include=['float64']).columns

    if len(numeric_cols) > 0:
        imputer = KNNImputer(n_neighbors=n_neighbors)
        weather_df[numeric_cols] = imputer.fit_transform(weather_df[numeric_cols])

    for col in categorical_cols:
        weather_df[col] = label_encoders[col].inverse_transform(weather_df[col].astype(int))

    return weather_df