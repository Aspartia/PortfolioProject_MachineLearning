
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import math

# --------------------------------------------------------------
# Chauvenet - Code from Dave Ebbelaar
# --------------------------------------------------------------

def mark_outliers_chauvenet(dataset, numeric_columns, C=2):

    dataset = dataset.copy()

    for col in numeric_columns:
        if col not in dataset.columns:
            print(f"Figyelmeztetés: '{col}' oszlop nem található a datasetben. Kihagyás...")
            continue 
        
        mean = dataset[col].mean()
        std = dataset[col].std()
        N = len(dataset.index)

        if N == 0 or std == 0:
            dataset[col + "_outlier"] = False
            continue

        criterion = 1.0 / (C * N)
        deviation = abs(dataset[col] - mean) / std

        if deviation.isnull().sum() > 0 or (deviation == np.inf).sum() > 0:
            print(f" Figyelmeztetés: '{col}' oszlopban NaN vagy végtelen érték található!")

        low = -deviation / np.sqrt(C)
        high = deviation / np.sqrt(C)

        prob = np.zeros(len(dataset))
        mask = np.zeros(len(dataset), dtype=bool) #Mask

        for i in range(len(dataset)):
            try:
                prob[i] = 1.0 - 0.5 * (scipy.special.erf(high.iloc[i]) - scipy.special.erf(low.iloc[i]))
                mask[i] = prob[i] < criterion
            except (IndexError, KeyError):
                print(f"Hiba: Index hiba történt a {i}-edik elemnél!")

        dataset[col + "_outlier"] = mask

        n_outliers = dataset[col + "_outlier"].sum()
        print(f"{n_outliers} outlier detektálva: {col}")

    return dataset



def remove_outliers_chauvenet(dataset_with_outliers, numeric_columns, remove_rows=False):

    cleaned_df = dataset_with_outliers.copy()

    for col in numeric_columns:
        outlier_col = col + "_outlier"

        if outlier_col not in cleaned_df.columns:
            print(f"Figyelmeztetés: '{outlier_col}' oszlop nem található! Ellenőrizd, hogy futott-e az outlier-jelölés!")
            continue  

        if remove_rows:
            cleaned_df = cleaned_df[~cleaned_df[outlier_col]]
        else:
            cleaned_df[col] = np.where(cleaned_df[outlier_col], np.nan, cleaned_df[col])

        n_outliers = cleaned_df[outlier_col].sum()
        print(f"{n_outliers} outlier eltávolítva/törölve: {col}")

    outlier_cols = [col + "_outlier" for col in numeric_columns if col + "_outlier" in cleaned_df.columns]
    if outlier_cols:
        cleaned_df.drop(columns=outlier_cols, errors="ignore", inplace=True)
    

    return cleaned_df



def remove_outliers_chauvenet2(dataset_with_outliers, numeric_columns,remove_rows=True):

    cleaned_df = dataset_with_outliers.copy()
    
    for col in numeric_columns:
        outlier_col = col + "_outlier"

        if outlier_col not in cleaned_df.columns:
            print(f" Figyelmeztetés: '{outlier_col}' nem található. Ellenőrizd, hogy futott-e az outlier-jelölés!")
            continue 

        #cleaned_df.loc[cleaned_df[outlier_col], col] = np.nan
        cleaned_df[col] = np.where(cleaned_df[outlier_col], np.nan, cleaned_df[col])

        n_outliers = cleaned_df[outlier_col].sum()
        print(f"{n_outliers} outlier eltávolítva: {col}")

    # # Az outlier oszlopok törlése, ha léteznek
    # outlier_cols = [col + "_outlier" for col in numeric_columns if col + "_outlier" in cleaned_df.columns]
    # if outlier_cols:
    #     cleaned_df.drop(columns=outlier_cols, errors="ignore", inplace=True)

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

