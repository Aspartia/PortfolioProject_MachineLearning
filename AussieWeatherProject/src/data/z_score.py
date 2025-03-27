import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def mark_outliers_zscore(dataset, numeric_columns, threshold=3):
    dataset = dataset.copy()

    for col in numeric_columns:
        mean = dataset[col].mean()
        std = dataset[col].std()

        dataset[col + "_outlier"] = abs((dataset[col] - mean) / std) > threshold

    return dataset

def remove_outliers_zscore(dataset_with_outliers, numeric_columns):
    cleaned_df = dataset_with_outliers.copy()

    for col in numeric_columns:
        outlier_col = col + "_outlier"
        cleaned_df.loc[cleaned_df[outlier_col], col] = np.nan 

    outlier_cols = [col + "_outlier" for col in numeric_columns]
    cleaned_df.drop(columns=outlier_cols, errors="ignore", inplace=True)

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

