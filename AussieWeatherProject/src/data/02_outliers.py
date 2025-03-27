import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# -------------------------------------------------------------------
# Import Data
# -------------------------------------------------------------------
weather_df = pd.read_pickle("../../data/interim/01_make_dataset.pkl")

# -------------------------------------------------------------------
# Info
# -------------------------------------------------------------------
weather_df.info()
weather_df.isna().sum()
weather_df.describe()
weather_df.columns

# -------------------------------------------------------------------
# Removing Outliers (IQR, Chauvenet, Z-score)
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# # IQR 
# -------------------------------------------------------------------

from IQR_outlier import plot_binary_outliers,  remove_outliers_iqr, mark_outliers_iqr

numeric_columns = weather_df.select_dtypes(include=['float64']).columns
#iqr outliers
dataset_with_outliers = mark_outliers_iqr(weather_df, numeric_columns)

#plot outliers
for col in numeric_columns:
    plot_binary_outliers(dataset=dataset_with_outliers, col=col, outlier_col=col + "_outlier", reset_index=True)
    #plt.savefig(f"../../reports/figures/outlier_{col}.png")
    
#remove
cleaned_df_IQR = remove_outliers_iqr(dataset_with_outliers, numeric_columns)
cleaned_df_IQR.isna().sum()
cleaned_df_IQR.info()
weather_df.isna().sum()

# -------------------------------------------------------------------
# # Chauvenet
# -------------------------------------------------------------------

from chauvenet import plot_binary_outliers, mark_outliers_chauvenet, remove_outliers_chauvenet

numeric_columns = weather_df.select_dtypes(include=['float64']).columns
dataset_with_outliers = mark_outliers_chauvenet(weather_df, numeric_columns)

# chauvenet
for col in numeric_columns:
    plot_binary_outliers(dataset= dataset_with_outliers, col= col, outlier_col= col + "_outlier", reset_index= True)

cleaned_df_chauv = remove_outliers_chauvenet(dataset_with_outliers, numeric_columns,remove_rows=False)
cleaned_df_chauv.isna().sum()
weather_df.isna().sum()

# -------------------------------------------------------------------
# # Z-score
# -------------------------------------------------------------------

from z_score import mark_outliers_zscore, remove_outliers_zscore, plot_binary_outliers

numeric_columns = weather_df.select_dtypes(include=['float64']).columns
dataset_with_outliers = mark_outliers_zscore(weather_df, numeric_columns, threshold=3)


for col in numeric_columns:
    plot_binary_outliers(dataset=dataset_with_outliers, col=col, outlier_col=col + "_outlier", reset_index=True)


cleaned_df_Zscore = remove_outliers_zscore(dataset_with_outliers, numeric_columns)

cleaned_df_Zscore.isna().sum()
weather_df.isna().sum()


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------
# Filling Missing Values (KNN, SimpleImputer, SimaFüggvény)  
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# # KNN Imputer
# -------------------------------------------------------------------
from knn_imputer import knn_impute

KNN_df_IQR = knn_impute(cleaned_df_IQR, n_neighbors=5) #
KNN_df_chauv = knn_impute(cleaned_df_chauv, n_neighbors=5) # gyors
KNN_df_Zscore = knn_impute(cleaned_df_Zscore, n_neighbors=5) #
KNN_df = knn_impute(weather_df, n_neighbors=5) #

KNN_df.info()
KNN_df_IQR.isnull().sum()
KNN_df_chauv.isnull().sum()
KNN_df_Zscore.isnull().sum()


# -------------------------------------------------------------------
# # SimpleImputer
# -------------------------------------------------------------------
from sklearn.impute import SimpleImputer
import numpy as np

def fill_missing_values_imputer(weather_df):
    for col in weather_df.columns:
        dtype = weather_df[col].dtype
        
        if dtype == 'object': 
            imputer = SimpleImputer(strategy='most_frequent')
            weather_df[col] = imputer.fit_transform(weather_df[[col]]).ravel() 
        
        elif np.issubdtype(dtype, np.number): 
            imputer = SimpleImputer(strategy='mean')
            weather_df[col] = imputer.fit_transform(weather_df[[col]]).ravel()
        
        elif dtype == 'bool': 
            imputer = SimpleImputer(strategy='most_frequent')
            weather_df[col] = imputer.fit_transform(weather_df[[col]].astype(int)).astype(bool) 

    return weather_df
    
    
imputer_df = fill_missing_values_imputer(weather_df)
imputer_df_IQR = fill_missing_values_imputer(cleaned_df_IQR)
imputer_df_chauv = fill_missing_values_imputer(cleaned_df_chauv)
imputer_df_Zscore = fill_missing_values_imputer(cleaned_df_Zscore)
imputer_df.info()  
imputer_df_IQR.isnull().sum()  
imputer_df_chauv.isnull().sum()  
imputer_df_Zscore.isnull().sum()  

# -------------------------------------------------------------------
# # Sima függvény (mean, mode, median)
# -------------------------------------------------------------------
def fill_missing_values(weather_df):
    for col in weather_df.columns:
        dtype = weather_df[col].dtype
        
    
        if dtype == 'object': 
            weather_df[col] = weather_df[col].fillna(weather_df[col].mode()[0])
        
        elif dtype in ['float64', 'int64']:  
            weather_df[col] = weather_df[col].fillna(weather_df[col].mean()) 
        
        
        elif dtype == 'bool': 
            weather_df[col] = weather_df[col].fillna(weather_df[col].mode()[0]) 
        
    return weather_df


function_df = fill_missing_values(weather_df)
function_df_IQR = fill_missing_values(cleaned_df_IQR)
function_df_chauv = fill_missing_values(cleaned_df_chauv)
function_df_Zscore = fill_missing_values(cleaned_df_Zscore)
function_df.info()    
function_df_IQR.isna().sum()    
function_df_chauv.isna().sum()   
function_df_Zscore.isna().sum()   

# -------------------------------------------------------------------
# Interpolate :(
# -------------------------------------------------------------------
# from interpolate import interpolate_numeric_columns

# interpolate_df = interpolate_numeric_columns(weather_df, method='linear') #jobb
# interpolate_df = interpolate_numeric_columns(weather_df, method='polynomial', order=2)
# interpolate_df.isna().sum()
    

# -------------------------------------------------------------------
# Visualization - Comparing Filled Data
# -------------------------------------------------------------------
#1
def compare_filling_methods_allInOne(original_df, knn_dfs, imputer_dfs, function_dfs, method_names, column):
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(original_df.index, original_df[column], "ko", label="Eredeti (hiányos) adatok", alpha=0.3)

    for knn_df, method in zip(knn_dfs, method_names):
        plt.plot(knn_df.index, knn_df[column], label=f"KNN - {method}", linestyle="dotted")

    for imp_df, method in zip(imputer_dfs, method_names):
        plt.plot(imp_df.index, imp_df[column], label=f"SimpleImputer - {method}", linestyle="dashed")

    for func_df, method in zip(function_dfs, method_names):
        plt.plot(func_df.index, func_df[column], label=f"Function-based - {method}", linestyle="solid")
    
    plt.legend()
    plt.xlabel("Időindex")
    plt.ylabel(column)
    plt.title(f"Hiányzó értékek kitöltésének összehasonlítása: {column}")
    #plt.savefig(f"../../reports/figures/Rainfall_allInOne.png")
    plt.show()

method_names = ["Alap", "IQR", "Chauvenet", "Z-score"]

compare_filling_methods_allInOne(
    original_df=weather_df,
    knn_dfs=[KNN_df, KNN_df_IQR, KNN_df_chauv, KNN_df_Zscore],
    imputer_dfs=[imputer_df, imputer_df_IQR, imputer_df_chauv, imputer_df_Zscore],
    function_dfs=[function_df, function_df_IQR, function_df_chauv, function_df_Zscore],
    method_names=method_names,
    column="Rainfall"  
)


#2
def compare_filling_methods(original_df, datasets_dict, method_names, column):
    fig, axes = plt.subplots(3, 4, figsize=(16, 12)) 
    axes = axes.flatten()  
    
    for i, (method, dfs) in enumerate(datasets_dict.items()):
        for j, (df, name) in enumerate(zip(dfs, method_names)):
            ax = axes[i * 4 + j] 
            ax.plot(original_df.index, original_df[column], "ko", label="Eredeti (hiányos) adatok", alpha=0.3)
            ax.plot(df.index, df[column], label=f"{method} - {name}", linestyle="dotted")
            ax.legend()
            ax.set_xlabel("Időindex")
            ax.set_ylabel(column)
            ax.set_title(f"{method} - {name}")

    plt.tight_layout()
    #plt.savefig(f"../../reports/figures/Pressure9am.png")
    plt.show()


datasets_dict = {
    "KNN": [KNN_df, KNN_df_IQR, KNN_df_chauv, KNN_df_Zscore],
    "Imputer": [imputer_df, imputer_df_IQR, imputer_df_chauv, imputer_df_Zscore],
    "Function": [function_df, function_df_IQR, function_df_chauv, function_df_Zscore]}

method_names = ["Alap", "IQR", "Chauvenet", "Z-score"]

compare_filling_methods(
    original_df=weather_df,
    datasets_dict=datasets_dict, 
    method_names=method_names,
    column="Pressure9am")




# -------------------------------------------------------------------
# Export data
# -------------------------------------------------------------------

KNN_df_IQR.to_pickle("../../data/interim/02_outliers_dataset_KNN_df_IQR.pkl")
KNN_df_chauv.to_pickle("../../data/interim/02_outliers_dataset_KNN_df_chauv.pkl")
KNN_df_Zscore.to_pickle("../../data/interim/02_outliers_dataset_KNN_df_Zscore.pkl")

imputer_df_IQR.to_pickle("../../data/interim/02_outliers_dataset_imputer_df_IQR.pkl")
imputer_df_chauv.to_pickle("../../data/interim/02_outliers_dataset_imputer_df_chauv.pkl")
imputer_df_Zscore.to_pickle("../../data/interim/02_outliers_dataset_imputer_df_Zscore.pkl")

function_df_IQR.to_pickle("../../data/interim/02_outliers_dataset_function_df_IQR.pkl")
function_df_chauv.to_pickle("../../data/interim/02_outliers_dataset_function_df_chauv.pkl")
function_df_Zscore.to_pickle("../../data/interim/02_outliers_dataset_function_df_Zscore.pkl")

