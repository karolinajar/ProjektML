from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import shap
import matplotlib.pyplot as plt
from variables import RANDOM_STATE, TEST_SIZE, PCA_COMPONENTS, TARGET_COLUMN, PLOT_FEATURE_IMPORTANCE
import pandas as pd
import numpy as np


def apply_pca(X, n_components=PCA_COMPONENTS):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    print(f"PCA zredukowało dane do {n_components} komponentów.")
    return X_pca


def balance_data(X, y, method="smote"):
    if method == "smote":
        smote = SMOTE(random_state=RANDOM_STATE)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        print("Dane zbalansowane za pomocą SMOTE.")
    elif method == "undersample":
        undersampler = RandomUnderSampler(random_state=RANDOM_STATE)
        X_balanced, y_balanced = undersampler.fit_resample(X, y)
        print("Dane zbalansowane za pomocą undersampling.")
    else:
        raise ValueError("Nieobsługiwana metoda balansowania danych.")
    return X_balanced, y_balanced


def grid_search_model(X, y, model_type="random_forest"):
    if model_type == "random_forest":
        model = RandomForestClassifier(random_state=RANDOM_STATE)
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
    elif model_type == "logistic_regression":
        model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        param_grid = {
            "C": [0.1, 1, 10],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
        }
    elif model_type == "xgboost":
        model = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss")
        param_grid = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 10],
            "subsample": [0.8, 1.0],
        }
    else:
        raise ValueError("Nieobsługiwany typ modelu.")

    print(f"\nRozpoczynanie Grid Search dla modelu: {model_type.capitalize()}...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring="roc_auc", verbose=2, n_jobs=-1)
    grid_search.fit(X, y)

    print("\nNajlepsze hiperparametry:")
    print(grid_search.best_params_)
    print(f"Najlepszy wynik ROC AUC: {grid_search.best_score_}")

    return grid_search.best_estimator_


def plot_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
        plt.xlabel("Ważność cechy")
        plt.ylabel("Cechy")
        plt.title("Ważność cech")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    else:
        print("Model nie obsługuje metody `feature_importances_`.")


def interpret_with_shap(model, X_train):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    # Debugowanie
    print("\nDebugowanie danych SHAP:")
    print(f"Typ shap_values: {type(shap_values)}")
    print(f"Rozmiar shap_values.values: {shap_values.values.shape}")
    print(f"Typ expected_value: {type(explainer.expected_value)}")
    print(f"Zawartość expected_value: {explainer.expected_value}")

    # Wykres Summary Plot
    print("\n### Wykres Summary Plot dla SHAP ###")
    shap.summary_plot(shap_values, X_train)

    # Decision Plot
    print("\n### Wykres decyzji (Decision Plot): ###")
    try:
        shap.decision_plot(
            explainer.expected_value,
            shap_values.values[:100],  # Redukcja do 100 obserwacji dla przejrzystości
            X_train.iloc[:100]        # Dopasowanie do tych samych obserwacji
        )
    except Exception as e:
        print(f"Nie udało się wygenerować wykresu decyzji. Szczegóły błędu: {e}")


def train_model(df, model_type="random_forest", balance_method="smote", use_grid_search=False, use_pca=True):

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Balansowanie danych
    X_train_balanced, y_train_balanced = balance_data(X_train, y_train, method=balance_method)

    # Opcjonalna redukcja wymiarowości za pomocą PCA
    if use_pca and PCA_COMPONENTS:
        X_train_balanced = apply_pca(X_train_balanced, n_components=PCA_COMPONENTS)
        X_test = apply_pca(X_test, n_components=PCA_COMPONENTS)

    # Wybór modelu z opcjonalnym Grid Search
    if use_grid_search:
        model = grid_search_model(X_train_balanced, y_train_balanced, model_type=model_type)
    else:
        if model_type == "random_forest":
            model = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")
        elif model_type == "logistic_regression":
            model = LogisticRegression(random_state=RANDOM_STATE, class_weight="balanced")
        elif model_type == "xgboost":
            model = XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss')
        else:
            raise ValueError("Nieobsługiwany typ modelu.")
        model.fit(X_train_balanced, y_train_balanced)

    # Predykcja i ocena
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("\nWyniki klasyfikacji:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

    # Ważność cech (dla modeli bez PCA)
    if not use_pca and PLOT_FEATURE_IMPORTANCE and hasattr(model, "feature_importances_"):
        plot_feature_importance(model, X.columns)

    # SHAP Interpretacja
    interpret_with_shap(model, X_train)

    return model
