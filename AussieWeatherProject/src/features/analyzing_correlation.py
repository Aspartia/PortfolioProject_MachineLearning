import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy.stats import shapiro, pearsonr, spearmanr, ttest_ind, ks_2samp, f_oneway
import scipy.stats as stats


# -------------------------------------------------------------------
# Eloszlás
# -------------------------------------------------------------------

def compare_numeric_features_with_target(weather_df, target_col="RainTomorrow"):

    df = weather_df.copy()
    
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns

    if len(categorical_cols) > 0:
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str)) 
            label_encoders[col] = le 


    df['RainToday'] = df['RainToday'].replace({'Yes': 1, 'No': 0})
    df['RainTomorrow'] = df['RainTomorrow'].replace({'Yes': 1, 'No': 0})

    # Numerikus oszlopok kiválasztása
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    significant_features = [] 

    for col in numeric_columns:
        if col == target_col:
            continue  

        rainy_days = df[df[target_col] == 1][col].dropna()
        dry_days = df[df[target_col] == 0][col].dropna()

        stat_ttest, p_value_ttest = ttest_ind(rainy_days, dry_days, equal_var=False)
        stat_ks, p_value_ks = ks_2samp(rainy_days, dry_days)
        #stat, p_value = shapiro(rainy_days, dry_days)
        stat, p = f_oneway(rainy_days, dry_days)

        print(f"\n Vizsgált oszlop: {col}")
        print(f"  - T-próba statisztika: {stat_ttest:.4f}, p-érték: {p_value_ttest:.4e}")
        print(f"  - Kolmogorov-Smirnov teszt statisztika: {stat_ks:.4f}, p-érték: {p_value_ks:.4e}")
        #print(f"Shapiro-Wilk p-érték: {p_value:.5f}")
        print(f"ANOVA p-érték: {p:.5f}")
        
    
        if p_value_ttest < 0.05 and stat_ks >= 0.2:
            significant_features.append(col)

    
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

        sns.histplot(dry_days, kde=True, color='orange', ax=axes[0])
        axes[0].set_title(f'{col} - Száraz napok')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Gyakoriság")

        sns.histplot(rainy_days, kde=True, color='blue', ax=axes[1])
        axes[1].set_title(f'{col} - Esős napok')
        axes[1].set_xlabel(col)
        axes[1].set_ylabel("Gyakoriság")

        plt.suptitle(f'{col} és {target_col} kapcsolata', fontsize=14)
        plt.tight_layout()
        plt.show()

    for col in categorical_cols:
        df[col] = label_encoders[col].inverse_transform(df[col].astype(int))

    if significant_features:
        print("\n A következő változók jó prediktorok lehetnek a célváltozóhoz (p < 0.05 és KS >= 0.2):")
        print(", ".join(significant_features))
        
    else:
        print("\n Nem találtunk szignifikáns prediktort a célváltozóra!")
        
        
# -------------------------------------------------------------------
# Multikollinearitás - Only Between Input Labels 
# -------------------------------------------------------------------       

def analyze_multicollinearity(weather_df):
    df = weather_df.copy()

    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns

    if len(categorical_cols) > 0:
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str)) 
            label_encoders[col] = le 

    df['RainToday'] = df['RainToday'].replace({'Yes': 1, 'No': 0}).astype(float)

    numeric_columns = df.select_dtypes(include=[np.number]).columns

    if "RainTomorrow" in numeric_columns:
        numeric_columns = numeric_columns.drop("RainTomorrow")

    correlation_results = []
    
    for col1 in numeric_columns:
        for col2 in numeric_columns:
            if col1 != col2:
                pearson_corr, p_pearson = pearsonr(df[col1].dropna(), df[col2].dropna())
                spearman_corr, p_spearman = spearmanr(df[col1].dropna(), df[col2].dropna())

                correlation_results.append({
                    "Feature 1": col1,
                    "Feature 2": col2,
                    "Pearson Correlation": pearson_corr,
                    "Pearson p-value": p_pearson,
                    "Spearman Correlation": spearman_corr,
                    "Spearman p-value": p_spearman
                })

    correlation_df = pd.DataFrame(correlation_results)

    correlated_features = correlation_df[(correlation_df["Pearson Correlation"] > 0.8) & 
                                         (correlation_df["Pearson p-value"] < 0.05)]
    
    print("\n **Magasan korreláló változók (Pearson > 0.8, p < 0.05):**")
    print(correlated_features[["Feature 1", "Feature 2", "Pearson Correlation"]])

    plt.figure(figsize=(20, 15))
    sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Teljes korrelációs mátrix az összes numerikus változóra")
    plt.show()

    # sns.pairplot(df[numeric_columns], diag_kind="kde", hue='RainToday')
    # plt.savefig(f"../../reports/figures/Korrelációs Pairplot.png")
    # plt.show()

    for col in categorical_cols:
        df[col] = label_encoders[col].inverse_transform(df[col].astype(int))
        
# -------------------------------------------------------------------
# Linearity & Correlation (Target and Input Labels )
# -------------------------------------------------------------------       

def analyze_linearity_correlation(weather_df, target_label="RainTomorrow"):

    df = weather_df.copy()
    
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    ignore_cols = ["Location"] 
    categorical_cols = [col for col in categorical_cols if col not in ignore_cols]

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str)) 
        label_encoders[col] = le 

    df['RainToday'] = df['RainToday'].replace({'Yes': 1, 'No': 0}).astype(float)
    df['RainTomorrow'] = df['RainTomorrow'].replace({'Yes': 1, 'No': 0}).astype(float)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col != target_label] 

    correlation_results = {}
    linear_columns = [] 

    print("\n**Korrelációs eredmények:**")
    
    for col in numeric_columns:
        stat, p_value = shapiro(df[col].dropna())

        if p_value > 0.05: 
            corr, corr_p_value = pearsonr(df[col].dropna(), df[target_label].dropna())
            method = "Pearson"
        else: 
            corr, corr_p_value = spearmanr(df[col].dropna(), df[target_label].dropna())
            method = "Spearman"

        correlation_results[col] = {
            "Correlation": corr, 
            "Corr_p_value": corr_p_value, 
            "Method": method
        }

        print(f"{col}: {method} korreláció = {corr:.4f}, p-érték = {corr_p_value:.4e}")

        if method == "Pearson" and corr_p_value < 0.05:
            linear_columns.append(col)

    correlation_df = pd.DataFrame(correlation_results).T.sort_values(by="Correlation", ascending=False)

    plt.figure(figsize=(12, 6))
    sns.heatmap(correlation_df[["Correlation"]].astype(float), annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title(f"Correlation of Features with {target_label} (Pearson/Spearman)")
    plt.show()

    # fig, axes = plt.subplots(len(numeric_columns), 2, figsize=(12, len(numeric_columns) * 3))
    # for i, col in enumerate(numeric_columns):
    #     sns.histplot(df[df[target_label] == 1][col], ax=axes[i, 0], kde=True, color='orange', label='Rain Tomorrow')
    #     sns.histplot(df[df[target_label] == 0][col], ax=axes[i, 1], kde=True, color='blue', label='No Rain Tomorrow')
    #     axes[i, 0].set_title(f"{col} Distribution (Rain Tomorrow)")
    #     axes[i, 1].set_title(f"{col} Distribution (No Rain Tomorrow)")
    #     axes[i, 0].legend()
    #     axes[i, 1].legend()

    # plt.tight_layout()
    # #plt.savefig(f"../../reports/figures/analyze_linearity_correlation_histo.png")
    # plt.show()

    for col in categorical_cols:
        df[col] = label_encoders[col].inverse_transform(df[col].astype(int))

    if linear_columns:
        print("\n **A következő oszlopok valószínűleg lineáris kapcsolatban állnak a célváltozóval:**")
        print(", ".join(linear_columns))
    else:
        print("\n **Nem találtunk erős lineáris kapcsolatú oszlopokat a célváltozóhoz.**")

    return correlation_df
    

def analyze_linearity_correlation2(weather_df, target_label="RainTomorrow"):
    df = weather_df.copy()
    
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str)) 
        label_encoders[col] = le 

    df['RainToday'] = df['RainToday'].replace({'Yes': 1, 'No': 0}).astype(float)
    df['RainTomorrow'] = df['RainTomorrow'].replace({'Yes': 1, 'No': 0}).astype(float)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col != target_label]

    correlation_results = {}
    linear_columns = [] 

    print("\n **Korrelációs eredmények:**")
    
    for col in numeric_columns:

        stat, p_value = shapiro(df[col].dropna())

        if p_value > 0.05: 
            corr, corr_p_value = pearsonr(df[col].dropna(), df[target_label].dropna())
            method = "Pearson"
        else:  
            corr, corr_p_value = spearmanr(df[col].dropna(), df[target_label].dropna())
            method = "Spearman"

        correlation_results[col] = {
            "Correlation": corr, 
            "Corr_p_value": corr_p_value, 
            "Method": method
        }

        print(f" {col}: {method} korreláció = {corr:.4f}, p-érték = {corr_p_value:.4e}")

        if method == "Pearson" and corr_p_value < 0.05:
            linear_columns.append(col)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.histplot(df[df[target_label] == 1][col], ax=axes[0], kde=True, color='blue', label='Rain Tomorrow')
        sns.histplot(df[df[target_label] == 0][col], ax=axes[0], kde=True, color='orange', label='No Rain Tomorrow')
        axes[0].set_title(f"{col} Distribution by {target_label}")
        axes[0].legend()

        stats.probplot(df[col].dropna(), dist="norm", plot=axes[1])
        axes[1].set_title(f"Q-Q plot - {col}")

        plt.tight_layout()
        plt.show()

    correlation_df = pd.DataFrame(correlation_results).T.sort_values(by="Correlation", ascending=False)

    for col in categorical_cols:
        df[col] = label_encoders[col].inverse_transform(df[col].astype(int))

    if linear_columns:
        print("\n**A következő oszlopok valószínűleg lineáris kapcsolatban állnak a célváltozóval:**")
        print(", ".join(linear_columns))
    else:
        print("\n**Nem találtunk erős lineáris kapcsolatú oszlopokat a célváltozóhoz.**")

    return correlation_df
