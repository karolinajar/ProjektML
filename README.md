# ProjektML

Projekt dotyczy przewidywania ryzyka wystąpienia udaru na podstawie cech demograficznych i medycznych pacjentów. Wykorzystano proces uczenia maszynowego, obejmujący wczytanie i przygotowanie danych, inżynierię danych, modelowanie oraz interpretację wyników. 

Dane zostały przetworzone, obejmując usuwanie wartości odstających, imputację braków, a także redukcję wymiarowości za pomocą PCA. Zaimplementowano trzy modele predykcyjne: Random Forest, Logistic Regression oraz XGBoost, z obsługą dostrajania hiperparametrów za pomocą GridSearchCV. Interpretacja wyników modelu została przeprowadzona za pomocą biblioteki SHAP, co pozwoliło na zrozumienie wpływu poszczególnych cech na decyzje modelu. Dodatkowo projekt zawiera zestaw wizualizacji, takich jak macierze korelacji, histogramy, wykresy pudełkowe i wykresy PCA.


main.py: Główna funkcja zarządzająca procesem projektu, od wczytania danych po modelowanie i interpretację wyników z obsługą wizualizacji.

data_processing.py: Zawiera funkcje do wczytywania, czyszczenia, przetwarzania danych, tworzenia nowych zmiennych oraz redukcji wymiarowości za pomocą PCA.

modeling.py: Implementuje modele predykcyjne (Random Forest, Logistic Regression, XGBoost) z obsługą GridSearchCV oraz interpretacji wyników za pomocą SHAP.

visualizations.py: Zawiera funkcje do generowania wizualizacji danych i wyników, w tym wykresów PCA, rozkładu klas i analiz korelacji.

variables.py: Zawiera wszystkie parametry konfiguracyjne projektu, takie jak ścieżki do plików, ustawienia modeli, progi korelacji i wartości wariancji.
