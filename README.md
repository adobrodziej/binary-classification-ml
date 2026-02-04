# Klasyfikacja binarna – pipeline ML (scikit-learn)

Projekt prezentuje prostą, ale profesjonalną implementację klasyfikacji binarnej
z wykorzystaniem biblioteki **scikit-learn** oraz podejścia pipeline.

Celem projektu jest pokazanie kompletnego procesu:
od wczytania danych, przez preprocessing i trening modelu,
po ewaluację i zapis wytrenowanego modelu.

---

## Dane

Wykorzystano publiczny zbiór danych **Breast Cancer Wisconsin**
dostępny w bibliotece `sklearn.datasets`.

- problem klasyfikacji binarnej (2 klasy)
- dane numeryczne
- brak ograniczeń licencyjnych

Dzięki temu projekt nie wymaga dołączania danych do repozytorium.

---

## Metodyka

- podział danych na zbiory treningowy i testowy (z zachowaniem proporcji klas)
- preprocessing w postaci **pipeline**
- możliwość wyboru modelu:
  - Logistic Regression
  - Random Forest
- ewaluacja jakości predykcji z użyciem standardowych metryk
- zapis wytrenowanego modelu oraz metryk do plików

---

## Metryki

Dla zbioru testowego obliczane są m.in.:

- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC
- Confusion Matrix

Metryki zapisywane są do pliku `metrics.json`.

---

## Technologie

- Python
- scikit-learn
- NumPy
- joblib

---

## Uruchomienie

1. (Opcjonalnie) utworzenie środowiska:
```bash
pip install scikit-learn numpy joblib
```

2. Uruchomienie z domyślnym modelem (Logistic Regression):
```bash
python binary_classification.py
```

3. Uruchomienie z innym modelem:
```bash
python binary_classification.py --model rf
