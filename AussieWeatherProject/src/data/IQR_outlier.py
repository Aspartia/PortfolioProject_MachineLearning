
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# --------------------------------------------------------------
# I: IQR
# --------------------------------------------------------------

def mark_outliers_iqr(weather_df, numeric_columns):
    weather_df = weather_df.copy()
    for col in numeric_columns:
        Q1 = weather_df[col].quantile(0.25) # 25%-os kvartilis (alsó kvartilis)
        Q3 = weather_df[col].quantile(0.75) # 75%-os kvartilis (felső kvartilis)
        IQR = Q3 - Q1 # Interkvartilis tartomány

        if col == "Rainfall":
            weather_df[col + "_outlier"] = False  
            continue 
        else:
            lower_bound = Q1 - 2.5 * IQR
            upper_bound = Q3 + 1.5 * IQR # 1.5 - ha csökkentem többet jelöl outliernek

        weather_df[col + "_outlier"] = (weather_df[col] < lower_bound) | (weather_df[col] > upper_bound)
    return weather_df


def remove_outliers_iqr(dataset_with_outliers, numeric_columns): #dataset, cols
    cleaned_df = dataset_with_outliers.copy() #dataset.copy
    for col in numeric_columns:
        cleaned_df.loc[cleaned_df[col + "_outlier"], col] = np.nan
        
        
    outlier_cols = [col + "_outlier" for col in numeric_columns]
    cleaned_df.drop(columns=outlier_cols, errors="ignore", inplace =True)
    
    return cleaned_df


def plot_binary_outliers(dataset, col, outlier_col, reset_index=True):
    """# Style from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py"""
    
    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])  
    dataset[outlier_col] = dataset[outlier_col].astype("bool")  

    if reset_index:
        dataset = dataset.reset_index(drop=True) 

    fig, ax = plt.subplots(figsize=(10, 5))

    plt.xlabel("Minta indexek")
    plt.ylabel("Értékek")

   
    ax.plot(dataset.index[~dataset[outlier_col]], dataset[col][~dataset[outlier_col]], "bo", label="Nem outlier")

    ax.plot(dataset.index[dataset[outlier_col]], dataset[col][dataset[outlier_col]], "ro", label="Outlier")

    plt.legend(loc="upper center", ncol=2, fancybox=True, shadow=True)
    plt.title(f"Outlierek megjelenítése - {col}")
    plt.show()


