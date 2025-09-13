#!/usr/bin/env python3
"""
Quantum Kernel SVM (QKSVM) — Minimal Working Example
----------------------------------------------------
A compact end-to-end demo of a quantum machine learning workflow using Qiskit.

What it does:
  1) Generates a non-linear 2D dataset (two moons).
  2) Embeds features into a quantum state via a ZZFeatureMap.
  3) Evaluates a quantum kernel on a simulator.
  4) Trains an SVM with the precomputed quantum kernel.
  5) Reports accuracy on a test split.

Usage:
  python quantum_kernel_svm.py --samples 400 --noise 0.2 --shots 1024

Requirements (install any that are missing):
  pip install "qiskit>=1.0" qiskit-aer qiskit-machine-learning scikit-learn numpy

Notes:
  - This script uses the Aer statevector/sampler backend when available.
  - If Qiskit or Aer are not installed, the script will print actionable hints.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np

# Scikit-learn for data and SVM
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Try-import Qiskit components with helpful errors
try:
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_aer import Aer
    from qiskit_machine_learning.kernels import QuantumKernel
except Exception as e:
    print("[ERROR] Qiskit (or a needed addon) is not available.\n"
          "Please install dependencies:\n"
          "  pip install \"qiskit>=1.0\" qiskit-aer qiskit-machine-learning\n"
          f"Details: {e}")
    sys.exit(1)


@dataclass
class Config:
    samples: int = 400
    test_size: float = 0.25
    noise: float = 0.2
    shots: int | None = 1024  # None -> statevector (exact); int -> sampler (shots)
    seed: int = 42
    reps: int = 2  # depth of ZZFeatureMap


def build_quantum_kernel(num_features: int, cfg: Config) -> QuantumKernel:
    """Create a QuantumKernel with a ZZFeatureMap and an Aer backend."""
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=cfg.reps)

    # Choose backend: statevector if shots is None; otherwise a shot-based sampler
    backend = Aer.get_backend("statevector_simulator") if cfg.shots is None else Aer.get_backend("qasm_simulator")

    if cfg.shots is None:
        # Exact kernel via statevector (fast, noiseless)
        qkernel = QuantumKernel(feature_map=feature_map, quantum_instance=backend)
    else:
        # Shot-based execution
        from qiskit.utils import QuantumInstance  # available for compatibility
        qinst = QuantumInstance(backend=backend, shots=cfg.shots, seed_simulator=cfg.seed, seed_transpiler=cfg.seed)
        qkernel = QuantumKernel(feature_map=feature_map, quantum_instance=qinst)

    return qkernel


def main(cfg: Config) -> int:
    rng = np.random.RandomState(cfg.seed)

    # 1) Data
    X, y = make_moons(n_samples=cfg.samples, noise=cfg.noise, random_state=cfg.seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y
    )

    # 2) Quantum Kernel
    qkernel = build_quantum_kernel(num_features=X.shape[1], cfg=cfg)

    print("[INFO] Evaluating quantum kernel (train)...")
    K_train = qkernel.evaluate(X_train, X_train)

    print("[INFO] Evaluating quantum kernel (test vs train)...")
    K_test = qkernel.evaluate(X_test, X_train)

    # 3) SVM with precomputed kernel
    clf = SVC(kernel="precomputed", probability=False, C=1.0, random_state=cfg.seed)
    clf.fit(K_train, y_train)

    # 4) Evaluate
    y_pred = clf.predict(K_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n=== Results ===")
    print(f"Samples: {cfg.samples} | Noise: {cfg.noise} | Shots: {cfg.shots}")
    print(f"Test Accuracy: {acc:.4f}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Kernel SVM (QKSVM) with Qiskit")
    parser.add_argument("--samples", type=int, default=400, help="Total samples for the moons dataset")
    parser.add_argument("--noise", type=float, default=0.2, help="Noise level for the moons dataset")
    parser.add_argument("--test-size", type=float, default=0.25, help="Test split fraction")
    parser.add_argument("--shots", type=int, default=1024, help="Number of shots; use 0 for statevector (exact)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--reps", type=int, default=2, help="Depth (reps) of ZZFeatureMap")
    args = parser.parse_args()

    cfg = Config(
        samples=args.samples,
        noise=args.noise,
        test_size=args.test_size,
        shots=None if int(args.shots) == 0 else int(args.shots),
        seed=args.seed,
        reps=args.reps,
    )

    try:
        sys.exit(main(cfg))
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        sys.exit(130)
