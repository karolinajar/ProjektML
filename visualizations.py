from pathlib import Path
from data_processing import (
    load_data,
    drop_unnecessary_columns,
    handle_missing_data,
    remove_outliers,
    remove_highly_correlated_features,
    encode_categorical_data,
    create_new_features,
    apply_pca,
)
import matplotlib.pyplot as plt
import seaborn as sns
from variables import PCA_COMPONENTS

#wykres słupkowy dla zmiennej docelowej
def visualize_target_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='stroke', hue='stroke', palette="Set2", legend=False)
    plt.title('Rozkład klasy docelowej')
    plt.xlabel('Stroke (0: Brak, 1: Wystąpił)')
    plt.ylabel('Liczba wystąpień')
    plt.show()

#wykres wyjaśnionej wariancji dla PCA
def visualize_pca_variance(pca):
    plt.figure(figsize=(8, 5))
    explained_variance = pca.explained_variance_ratio_
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
    plt.title('Wariancja wyjaśniona przez komponenty PCA')
    plt.xlabel('Komponenty PCA')
    plt.ylabel('Procent wyjaśnionej wariancji')
    plt.show()

#macierz korelacji
def visualize_correlations(df):
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Macierz korelacji')
    plt.show()

#wykresy rozkładów zmiennych numerycznych
def visualize_numerical_distributions(df):
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x=col, kde=True, bins=30)
        plt.title(f'Rozkład zmiennej: {col}')
        plt.xlabel(col)
        plt.ylabel('Częstość')
        plt.show()


def visualize_boxplots_with_adjustments(df):
    numerical_columns = [col for col in df.columns if col != 'stroke']

    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numerical_columns, 1):
        plt.subplot(3, 4, i)
        sns.boxplot(data=df, x=col)
        plt.title(f"Wykres pudełkowy: {col}")
    plt.tight_layout()
    plt.show()

#wykresy dla nowych cech
def visualize_new_features(df):
    if 'age_bmi_interaction' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x='age_bmi_interaction', kde=True, bins=30)
        plt.title('Rozkład zmiennej: age_bmi_interaction')
        plt.xlabel('age_bmi_interaction')
        plt.ylabel('Częstość')
        plt.show()

    if 'log_avg_glucose_level' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x='log_avg_glucose_level', kde=True, bins=30)
        plt.title('Rozkład zmiennej: log_avg_glucose_level')
        plt.xlabel('log_avg_glucose_level')
        plt.ylabel('Częstość')
        plt.show()

#histogramy dla komponentów PCA
def visualize_pca_distributions(df, pca_columns):
    for col in pca_columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x=col, kde=True, bins=30, color='blue', edgecolor='black')
        plt.title(f'Rozkład wartości dla {col}')
        plt.xlabel(f'Wartości {col}')
        plt.ylabel('Częstość')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()


def generate_visualizations(df, pca=None):
    visualize_target_distribution(df)
    visualize_correlations(df)
    visualize_numerical_distributions(df)
    visualize_boxplots_with_adjustments(df)
    visualize_new_features(df)

    if pca:
        pca_columns = [f'PCA_{i + 1}' for i in range(pca.n_components_)]
        visualize_pca_distributions(df, pca_columns)
        visualize_pca_variance(pca)


def main():
    path_to_file = Path.cwd().joinpath('datasets', 'stroke.csv')



    df = load_data(path_to_file)
    df = drop_unnecessary_columns(df, ['id'])
    df = handle_missing_data(df, strategy='mean')
    df = remove_outliers(df, columns=['age', 'bmi', 'avg_glucose_level'])
    df = create_new_features(df)
    df = encode_categorical_data(df)
    df = remove_highly_correlated_features(df, threshold=0.9)

    #redukcja wymiarowości
    pca = None
    if PCA_COMPONENTS:
        df, pca = apply_pca(df, target_column='stroke', n_components=PCA_COMPONENTS)


    generate_visualizations(df, pca)


if __name__ == "__main__":
    main()
