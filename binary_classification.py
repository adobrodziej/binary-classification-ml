"""
binary_classification.py

Cel projektu:
- Pokazać prosty, ale profesjonalny pipeline klasyfikacji binarnej:
  dane -> podział train/test -> preprocessing -> model -> metryki -> zapis modelu

Jak uruchomić:
    python binary_classification.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# scikit-learn: dataset + modelowanie + metryki + pipeline
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Do zapisu modelu 
import joblib


# =========================
# 1) Konfiguracja projektu
# =========================

@dataclass(frozen=True)
class Config:
    """
    Prosta konfiguracja w jednym miejscu.
    Dzięki temu łatwo zmieniać parametry bez grzebania po całym pliku.
    """
    test_size: float = 0.2
    random_state: int = 42
    output_dir: str = "artifacts"  # tu zapisujemy model i metryki


# =========================
# 2) Pobranie danych
# =========================

def load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Wczytujemy publiczny zbiór Breast Cancer Wisconsin (Binary classification).
    To klasyczny dataset: etykieta to 0/1.
    Zwracamy:
      X: cechy (numpy array)
      y: etykiety (numpy array)
      feature_names: nazwy cech (lista stringów)
    """
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target
    feature_names = dataset.feature_names.tolist()
    return X, y, feature_names


# =========================
# 3) Budowa modelu / pipeline
# =========================

def build_model(model_name: str, cfg: Config) -> Pipeline:
    """
    Budujemy pipeline:
    - StandardScaler (skalowanie cech) -> tylko dla modeli wrażliwych na skalę (np. logreg)
    - Model klasyfikacyjny

    Zwracamy Pipeline, który ma metodę .fit() i .predict().
    """
    model_name = model_name.lower()

    if model_name == "logreg":
        # Logistic Regression
        classifier = LogisticRegression(
            max_iter=2000,
            random_state=cfg.random_state,
            n_jobs=None,
        )

        pipeline = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("model", classifier),
        ])
        return pipeline

    if model_name == "rf":
        # Random Forest
        classifier = RandomForestClassifier(
            n_estimators=300,
            random_state=cfg.random_state,
            n_jobs=-1,
        )

        pipeline = Pipeline(steps=[
            ("model", classifier),
        ])
        return pipeline

    raise ValueError(f"Nieznany model: {model_name}. Wybierz: logreg lub rf.")


# =========================
# 4) Trening i ewaluacja
# =========================

def evaluate(model: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Liczymy standardowe metryki klasyfikacji:
    - accuracy
    - precision
    - recall
    - f1
    - roc_auc (jeśli model ma predict_proba)

    Zwracamy dict z metrykami, który łatwo zapisać do JSON.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
    }

    # ROC AUC wymaga prawdopodobieństw klasy pozytywnej
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    else:
        metrics["roc_auc"] = None

    # Dodatkowo: macierz pomyłek (confusion matrix)
    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def save_artifacts(model: Pipeline, metrics: dict, cfg: Config) -> None:
    """
    Zapisujemy:
    - wytrenowany model (joblib)
    - metryki (json)
    """
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.joblib"
    metrics_path = out_dir / "metrics.json"

    joblib.dump(model, model_path)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


# =========================
# 5) CLI (uruchamianie z parametrami)
# =========================

def parse_args() -> argparse.Namespace:
    """
    Dodajemy prosty interfejs:
    - wybór modelu
    Dzięki temu rekruter widzi, że potrafisz pisać kod "narzędziowy", a nie tylko notebooki.
    """
    parser = argparse.ArgumentParser(description="Binary classification with sklearn pipeline.")
    parser.add_argument(
        "--model",
        type=str,
        default="logreg",
        choices=["logreg", "rf"],
        help="Wybierz model: logreg (Logistic Regression) lub rf (Random Forest).",
    )
    return parser.parse_args()


# =========================
# 6) Main
# =========================

def main() -> None:
    cfg = Config()
    args = parse_args()

    # 1) Dane
    X, y, feature_names = load_data()

    # 2) Podział train/test
    # stratify=y -> zachowuje proporcje klas w train i test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    # 3) Budowa pipeline
    model = build_model(args.model, cfg)

    # 4) Trening
    model.fit(X_train, y_train)

    # 5) Ewaluacja
    metrics = evaluate(model, X_test, y_test)

    # 6) Drukujemy czytelne podsumowanie
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Zbiór: Breast Cancer Wisconsin (sklearn.datasets)")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print("-" * 60)
    print("Metryki:")
    for k, v in metrics.items():
        if k == "confusion_matrix":
            continue
        print(f"  {k}: {v}")
    print("-" * 60)
    print("Confusion matrix:")
    print(np.array(metrics["confusion_matrix"]))
    print("=" * 60)

    # 7) Zapis modelu + metryk
    save_artifacts(model, metrics, cfg)
    print(f"Zapisano artefakty do folderu: {cfg.output_dir}/")


if __name__ == "__main__":
    main()
