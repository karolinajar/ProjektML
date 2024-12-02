import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from variables import THRESHOLD_CORRELATION, THRESHOLD_VARIANCE
from pathlib import Path as p

def load_data(path):
    path_to_file = p.cwd().joinpath('datasets', 'stroke.csv')
    df = pd.read_csv(path_to_file)
    return df


def drop_unnecessary_columns(df, columns_to_drop):
    print(f"\nUsuwanie kolumn: {columns_to_drop}")
    df = df.drop(columns=columns_to_drop, errors='ignore')
    return df


def handle_missing_data(df, strategy='mean'):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Imputacja dla kolumn numerycznych
    if not numerical_cols.empty:
        num_imputer = SimpleImputer(strategy=strategy)
        df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

    # Imputacja dla kolumn kategorycznych
    if not categorical_cols.empty:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    return df


def remove_outliers(df, columns, threshold=1.5):
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    print("Wartości odstające zostały usunięte.")
    return df


def remove_highly_correlated_features(df, threshold=0.9):
    correlation_matrix = df.corr()
    correlated_features = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
    df = df.drop(columns=correlated_features, errors='ignore')
    print(f"Usunięto cechy o korelacji > {threshold}: {correlated_features}")
    return df


def encode_categorical_data(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    print(f"Kodowanie zmiennych kategorycznych: {list(categorical_columns)}")
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    return df


def select_features_by_variance(df, threshold=0.01):
    selector = VarianceThreshold(threshold)
    df_selected = selector.fit_transform(df)
    selected_columns = df.columns[selector.get_support()]
    print(f"Zachowano cechy o wariancji > {threshold}: {list(selected_columns)}")
    return pd.DataFrame(df_selected, columns=selected_columns)


def create_new_features(df):
    # Interakcje między cechami
    if 'age' in df.columns and 'bmi' in df.columns:
        df['age_bmi_interaction'] = df['age'] * df['bmi']

    # Logarytmiczne przekształcenie dla zmiennych numerycznych
    if 'avg_glucose_level' in df.columns:
        df['log_avg_glucose_level'] = df['avg_glucose_level'].apply(lambda x: np.log(x + 1))  # +1 aby uniknąć log(0)

    # Grupowanie kategorii w zmiennej
    if 'smoking_status' in df.columns:
        df['smoking_status_grouped'] = df['smoking_status'].replace({
            'formerly smoked': 'smoked',
            'never smoked': 'non-smoker',
            'smokes': 'smoked'
        })

    print("Nowe zmienne zostały dodane.")
    return df


def apply_pca(df, target_column, n_components=0.95):
    features = df.drop(columns=[target_column])
    target = df[target_column]

    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)

    # Przekształcenie zredukowanych cech z powrotem do DataFrame
    pca_columns = [f'PCA_{i + 1}' for i in range(reduced_features.shape[1])]
    reduced_df = pd.DataFrame(reduced_features, columns=pca_columns)

    # Dodanie kolumny docelowej
    reduced_df[target_column] = target.reset_index(drop=True)
    print(
        f"PCA zredukowało dane do {reduced_features.shape[1]} komponentów, zachowując {n_components * 100}% wariancji."
    )

    # Analiza ładunków PCA
    analyze_pca_loadings(pca, features.columns)

    return reduced_df, pca


def analyze_pca_loadings(pca, feature_names):
    loadings = pd.DataFrame(pca.components_, columns=feature_names)
    loadings.index = [f'PCA_{i+1}' for i in range(len(pca.components_))]
    print("\n### Wagi cech dla komponentów PCA ###")
    print(loadings)
    return loadings
