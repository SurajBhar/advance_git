#!/usr/bin/env python3
"""
Quantum Kernel SVM (QKSVM) — Extended ML Lifecycle + Hyperparameter Tuning
--------------------------------------------------------------------------
End-to-end ML lifecycle with quantum kernel methods.

Features:
  - Configurable dataset generation & preprocessing
  - Quantum kernel embedding with ZZFeatureMap
  - SVM training with precomputed quantum kernel
  - Evaluation metrics (accuracy, confusion matrix)
  - Visualization (decision boundary & kernel heatmap)
  - Experiment logging (JSON + CSV)
  - Model persistence (pickle save/load)
  - Hyperparameter tuning (grid search over C, reps, shots)

Usage:
  python quantum_kernel_svm.py train --samples 400 --noise 0.2
  python quantum_kernel_svm.py eval --model-path results/model.pkl
  python quantum_kernel_svm.py plot --model-path results/model.pkl
  python quantum_kernel_svm.py tune --samples 200 --noise 0.3

Requirements:
  pip install "qiskit>=1.0" qiskit-aer qiskit-machine-learning scikit-learn matplotlib seaborn numpy pandas
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Qiskit imports
try:
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_aer import Aer
    from qiskit_machine_learning.kernels import QuantumKernel
    from qiskit.utils import QuantumInstance
except Exception as e:
    print("[ERROR] Missing Qiskit packages. Install with:")
    print("  pip install \"qiskit>=1.0\" qiskit-aer qiskit-machine-learning")
    sys.exit(1)


# ------------------------
# Config dataclass
# ------------------------
@dataclass
class Config:
    samples: int = 400
    test_size: float = 0.25
    noise: float = 0.2
    shots: int | None = 1024
    seed: int = 42
    reps: int = 2
    scale: bool = True
    outdir: str = "results"


# ------------------------
# Data pipeline
# ------------------------
def load_data(cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_moons(n_samples=cfg.samples, noise=cfg.noise, random_state=cfg.seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed
    )

    if cfg.scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


# ------------------------
# Quantum kernel
# ------------------------
def build_quantum_kernel(num_features: int, cfg: Config) -> QuantumKernel:
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=cfg.reps)
    backend = Aer.get_backend("statevector_simulator") if cfg.shots is None else Aer.get_backend("qasm_simulator")

    if cfg.shots is None:
        qkernel = QuantumKernel(feature_map=feature_map, quantum_instance=backend)
    else:
        qinst = QuantumInstance(backend=backend, shots=cfg.shots, seed_simulator=cfg.seed, seed_transpiler=cfg.seed)
        qkernel = QuantumKernel(feature_map=feature_map, quantum_instance=qinst)

    return qkernel


# ------------------------
# Training + Evaluation
# ------------------------
def train_and_eval(cfg: Config, C: float = 1.0):
    X_train, X_test, y_train, y_test = load_data(cfg)
    qkernel = build_quantum_kernel(num_features=X_train.shape[1], cfg=cfg)

    print(f"[INFO] Training SVM with C={C}, reps={cfg.reps}, shots={cfg.shots}")
    K_train = qkernel.evaluate(X_train, X_train)
    K_test = qkernel.evaluate(X_test, X_train)

    clf = SVC(kernel="precomputed", C=C, random_state=cfg.seed)
    clf.fit(K_train, y_train)

    y_pred = clf.predict(K_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    Path(cfg.outdir).mkdir(exist_ok=True, parents=True)
    metrics = {"accuracy": acc, "confusion_matrix": cm.tolist(), "C": C, **asdict(cfg)}
    with open(Path(cfg.outdir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(Path(cfg.outdir) / "model.pkl", "wb") as f:
        pickle.dump({"clf": clf, "qkernel": qkernel, "X_train": X_train, "y_train": y_train}, f)

    return clf, qkernel, X_train, X_test, y_train, y_test, acc


# ------------------------
# Plotting helpers
# ------------------------
def plot_results(clf, qkernel, X_train, X_test, y_train, y_test, outdir="results"):
    print("[INFO] Plotting results...")
    xx, yy = np.meshgrid(np.linspace(X_train[:, 0].min() - 1, X_train[:, 0].max() + 1, 200),
                         np.linspace(X_train[:, 1].min() - 1, X_train[:, 1].max() + 1, 200))
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    K_grid = qkernel.evaluate(X_grid, X_train)
    Z = clf.predict(K_grid).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.4)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker="o", edgecolor="k", label="Train")
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker="s", edgecolor="k", label="Test")
    plt.title("Quantum Kernel SVM Decision Boundary")
    plt.legend()
    plt.savefig(Path(outdir) / "decision_boundary.png")
    plt.close()

    y_pred = clf.predict(qkernel.evaluate(X_test, X_train))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(Path(outdir) / "confusion_matrix.png")
    plt.close()


# ------------------------
# Hyperparameter tuning
# ------------------------
def tune_hyperparams(cfg: Config, C_grid=[0.1, 1.0, 10], reps_grid=[1, 2, 3], shots_grid=[0, 1024]):
    results = []
    for C in C_grid:
        for reps in reps_grid:
            for shots in shots_grid:
                run_cfg = Config(
                    samples=cfg.samples,
                    noise=cfg.noise,
                    test_size=cfg.test_size,
                    shots=None if shots == 0 else shots,
                    seed=cfg.seed,
                    reps=reps,
                    scale=cfg.scale,
                    outdir=cfg.outdir,
                )
                _, _, _, _, _, _, acc = train_and_eval(run_cfg, C=C)
                results.append({"C": C, "reps": reps, "shots": shots, "accuracy": acc})

    df = pd.DataFrame(results)
    df.to_csv(Path(cfg.outdir) / "tuning_results.csv", index=False)
    best = df.loc[df["accuracy"].idxmax()]
    print("\n[RESULT] Best configuration:")
    print(best)
    return df


# ------------------------
# CLI
# ------------------------
def main():
    parser = argparse.ArgumentParser(description="Quantum Kernel SVM with ML lifecycle + tuning")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--samples", type=int, default=400)
    train_parser.add_argument("--noise", type=float, default=0.2)
    train_parser.add_argument("--test-size", type=float, default=0.25)
    train_parser.add_argument("--shots", type=int, default=1024)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--reps", type=int, default=2)
    train_parser.add_argument("--scale", action="store_true")
    train_parser.add_argument("--outdir", type=str, default="results")

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--model-path", type=str, required=True)

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("--model-path", type=str, required=True)

    tune_parser = subparsers.add_parser("tune")
    tune_parser.add_argument("--samples", type=int, default=200)
    tune_parser.add_argument("--noise", type=float, default=0.3)
    tune_parser.add_argument("--test-size", type=float, default=0.25)
    tune_parser.add_argument("--seed", type=int, default=42)
    tune_parser.add_argument("--outdir", type=str, default="results")

    args = parser.parse_args()

    if args.command == "train":
        cfg = Config(
            samples=args.samples,
            noise=args.noise,
            test_size=args.test_size,
            shots=None if int(args.shots) == 0 else int(args.shots),
            seed=args.seed,
            reps=args.reps,
            scale=args.scale,
            outdir=args.outdir,
        )
        clf, qkernel, X_train, X_test, y_train, y_test, _ = train_and_eval(cfg)
        plot_results(clf, qkernel, X_train, X_test, y_train, y_test, outdir=cfg.outdir)

    elif args.command == "eval":
        with open(args.model_path, "rb") as f:
            bundle = pickle.load(f)
        print("[INFO] Model loaded:", bundle.keys())

    elif args.command == "plot":
        with open(args.model_path, "rb") as f:
            bundle = pickle.load(f)
        clf, qkernel, X_train, y_train = bundle["clf"], bundle["qkernel"], bundle["X_train"], bundle["y_train"]
        _, X_test, _, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
        plot_results(clf, qkernel, X_train, X_test, y_train, y_test, outdir="results")

    elif args.command == "tune":
        cfg = Config(
            samples=args.samples,
            noise=args.noise,
            test_size=args.test_size,
            seed=args.seed,
            outdir=args.outdir,
        )
        tune_hyperparams(cfg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
        sys.exit(130)
