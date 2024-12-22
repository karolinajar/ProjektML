from pathlib import Path as p
from variables import (
    PATH_TO_FILE, COLUMNS_TO_DROP, THRESHOLD_CORRELATION, THRESHOLD_VARIANCE,
    BALANCE_METHOD, MODEL_TYPE, TARGET_COLUMN, USE_GRID_SEARCH, PCA_COMPONENTS, INTERPRET_RESULTS
)
from data_processing import (
    load_data,
    drop_unnecessary_columns,
    handle_missing_data,
    remove_outliers,
    remove_highly_correlated_features,
    encode_categorical_data,
    select_features_by_variance,
    create_new_features,
    apply_pca
)
from modeling import train_model
import pandas as pd

results_path = p('results')
results_path.mkdir(parents=True, exist_ok=True)
def display_sample(df, message):
    print(message)
    print(df.head(), "\n")

#testtest
def count_missing_values(df):
    return df.isnull().sum()


def main(show_visualizations=False, interpret_results=True):

    path_to_file = p.cwd().joinpath('datasets', 'stroke.csv')

    df = pd.read_csv(path_to_file)

    df = load_data(path_to_file)
    print(f"Wczytano dane: {df.shape[0]} wierszy, {df.shape[1]} kolumn.")
    display_sample(df, "Przykładowe dane po wczytaniu:")

    # usuwanie niepotrzebnych kolumn
    df = drop_unnecessary_columns(df, COLUMNS_TO_DROP)
    print(f"Dane po usunięciu kolumn: {df.shape[0]} wierszy, {df.shape[1]} kolumn.")
    display_sample(df, "Przykładowe dane po usunięciu kolumn:")

    # obsługa braków danych
    missing_before = count_missing_values(df)
    print("Liczba braków danych przed imputacją:")
    print(missing_before)

    df = handle_missing_data(df, strategy='mean')
    missing_after = count_missing_values(df)
    print("\nLiczba braków danych po imputacji:")
    print(missing_after)


    # tworzenie nowych zmiennych
    df = create_new_features(df)
    display_sample(df, "Przykładowe dane po dodaniu nowych zmiennych:")

    # usuwanie wartości odstających
    df = remove_outliers(df, columns=['age', 'bmi', 'avg_glucose_level'])
    display_sample(df, "Przykładowe dane po usunięciu wartości odstających:")

    # kodowanie zmiennych kategorycznych
    df = encode_categorical_data(df)
    display_sample(df, "Przykładowe dane po kodowaniu zmiennych kategorycznych:")

    # selekcja cech na podstawie wariancji
    df = select_features_by_variance(df, THRESHOLD_VARIANCE)
    display_sample(df, "Przykładowe dane po selekcji cech na podstawie wariancji:")

    # usuwanie cech o wysokiej korelacji
    df = remove_highly_correlated_features(df, THRESHOLD_CORRELATION)
    display_sample(df, "Przykładowe dane po usunięciu cech o wysokiej korelacji:")

    # 9. Modelowanie z PCA
    print("\n### Modelowanie z PCA ###")
    train_model(df, model_type=MODEL_TYPE, balance_method=BALANCE_METHOD, use_grid_search=USE_GRID_SEARCH, use_pca=True)

    # 10. Modelowanie bez PCA
    print("\n### Modelowanie bez PCA ###")
    train_model(df, model_type=MODEL_TYPE, balance_method=BALANCE_METHOD, use_grid_search=USE_GRID_SEARCH, use_pca=False)

    print("testest")


if __name__ == "__main__":
    main(show_visualizations=False, interpret_results=INTERPRET_RESULTS)
