
import os
import kaggle 
from kaggle.api.kaggle_api_extended import KaggleApi
import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Download latest version from Kaggle
# -------------------------------------------------------------------

def download_kaggle_dataset(kaggle_path, local_path):
    api = KaggleApi()
    kaggle.api.authenticate()
    
    print("Elérhető fájlok a Kaggle datasetben:", kaggle.api.dataset_list_files(kaggle_path).files)
    kaggle.api.dataset_download_files(kaggle_path, path= local_path, unzip=True)

    files = os.listdir(local_path)
    print("Kicsomagolt fájlok:")
    for file in files:
        print(file)
    

kaggle_path = 'jsphyg/weather-dataset-rattle-package'
local_path = r'C:\Users\Évi\OneDrive\Asztali gép\Portfolio project\5. machine learning\AussieWeather\AussieWeatherProject\data\raw'

download_kaggle_dataset(kaggle_path,local_path)

# -------------------------------------------------------------------
# Import File & Basic Info
# -------------------------------------------------------------------
weather_df = pd.read_csv("../../data/raw/weatherAUS.csv") #*

weather_df.head()
weather_df.info()
weather_df.columns

# Count Missing Values
weather_df.isnull().sum()

# Count Duplicates
weather_df.duplicated().sum()
# weather_df.drop_duplicates()

# -------------------------------------------------------------------
# Date Conversations
# -------------------------------------------------------------------
weather_df['Date'] = pd.to_datetime(weather_df['Date']) #*

# -------------------------------------------------------------------
# Date to Index
# -------------------------------------------------------------------
#Sajnos nem egyediek az Date dátumértékei, ezért nem tehetők csak meg Indexnek
weather_df.value_counts().loc['2017-06-23'] # 49

# -------------------------------------------------------------------
# Dropping Missing TargetValues
# -------------------------------------------------------------------
weather_df = weather_df[~weather_df['RainTomorrow'].isna()] #*
weather_df = weather_df[~weather_df['RainToday'].isna()] 

# -------------------------------------------------------------------
# Convert Target to Bool 
# -------------------------------------------------------------------
# A hiányzó értékek elvetése a TargetLabelben: 

weather_df['RainToday'] = weather_df['RainToday'].map({'Yes': 1, 'No': 0}).astype(bool) #*
weather_df['RainTomorrow'] = weather_df['RainTomorrow'].map({'Yes': 1, 'No': 0}).astype(bool)

# -------------------------------------------------------------------
# Convert WindDirections
# ------------------------------------------------------------------- 
wind_directions = ['WindGustDir', 'WindDir9am', 'WindDir3pm']  #*
direction_mapping = {'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5,
                     'SE': 135, 'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225,
                     'WSW': 247.5, 'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5}

for col in wind_directions:
    weather_df[col] = weather_df[col].map(direction_mapping) 
    
# -------------------------------------------------------------------
# Resampling data - :(
# -------------------------------------------------------------------

# from weather_resampling import resample_weather_data
# weather_resampled = resample_weather_data(weather_df, freq='D')


# weather_resampled.isna().sum()
# weather_resampled.shape # (3525, 22)


# from weather_resampling_lo import group_by_date, resample_weather_data

# days = group_by_date(weather_df)
# weather_resampled = resample_weather_data(days)

# weather_resampled.isna().sum()
# weather_resampled.shape #(3436, 19)



# -------------------------------------------------------------------
# Export data
# -------------------------------------------------------------------
weather_df.to_pickle("../../data/interim/01_make_dataset.pkl")  #*

