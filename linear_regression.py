"""
Linear Regression - Educational Implementation
================================================
This program demonstrates Linear Regression from scratch using NumPy,
and compares it with scikit-learn's implementation.

Concepts covered:
  1. What is Linear Regression?
  2. Cost Function (Mean Squared Error)
  3. Gradient Descent Optimization
  4. Training and Prediction
  5. Model Evaluation (R², MSE, MAE)
  6. Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as SKLearnLinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────────
# STEP 1: Linear Regression from Scratch
# ──────────────────────────────────────────────

class LinearRegressionScratch:
    """
    Simple Linear Regression using Gradient Descent.

    Model:  y_pred = X @ weights + bias
    Loss:   MSE = (1/n) * sum((y_pred - y_true)²)

    Gradient Descent updates:
        weights -= lr * (dL/dw)
        bias    -= lr * (dL/db)
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.n_iterations):
            y_pred = X @ self.weights + self.bias

            # Compute gradients
            error = y_pred - y
            dw = (1 / n_samples) * X.T @ error
            db = (1 / n_samples) * np.sum(error)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias    -= self.lr * db

            # Record loss every 100 iterations
            if i % 100 == 0:
                loss = np.mean(error ** 2)
                self.loss_history.append(loss)

        return self

    def predict(self, X):
        return X @ self.weights + self.bias


# ──────────────────────────────────────────────
# STEP 2: Generate Synthetic Dataset
# ──────────────────────────────────────────────

def generate_dataset(n_samples=200, n_features=1, noise=20, random_state=42):
    """Create a synthetic regression dataset."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state
    )
    return X, y


# ──────────────────────────────────────────────
# STEP 3: Evaluation Metrics
# ──────────────────────────────────────────────

def evaluate(y_true, y_pred, label="Model"):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    print(f"\n{'─'*40}")
    print(f"  {label} Evaluation")
    print(f"{'─'*40}")
    print(f"  MSE  (Mean Squared Error)      : {mse:.4f}")
    print(f"  RMSE (Root Mean Squared Error) : {rmse:.4f}")
    print(f"  MAE  (Mean Absolute Error)     : {mae:.4f}")
    print(f"  R²   (Coefficient of Determ.)  : {r2:.4f}")
    print(f"{'─'*40}")
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


# ──────────────────────────────────────────────
# STEP 4: Visualizations
# ──────────────────────────────────────────────

def plot_all(X_train, X_test, y_train, y_test,
             scratch_preds, sklearn_preds, loss_history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Linear Regression — Educational Demo", fontsize=14, fontweight="bold")

    # --- Plot 1: Fit on Training Data ---
    ax = axes[0]
    x_range = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    ax.scatter(X_train, y_train, alpha=0.5, color="steelblue", label="Train data", s=20)
    ax.scatter(X_test, y_test, alpha=0.5, color="orange", label="Test data", s=20)
    ax.set_xlabel("Feature (X)")
    ax.set_ylabel("Target (y)")
    ax.set_title("Dataset")
    ax.legend()

    # --- Plot 2: Prediction Lines ---
    ax = axes[1]
    ax.scatter(X_test, y_test, alpha=0.5, color="gray", label="Actual", s=20)
    ax.scatter(X_test, scratch_preds, alpha=0.7, color="crimson",
               label="Scratch model", s=20, marker="x")
    ax.scatter(X_test, sklearn_preds, alpha=0.7, color="green",
               label="sklearn model", s=20, marker="+")
    ax.set_xlabel("Feature (X)")
    ax.set_ylabel("Predicted y")
    ax.set_title("Predictions vs Actual (Test Set)")
    ax.legend()

    # --- Plot 3: Training Loss Curve ---
    ax = axes[2]
    ax.plot(range(0, len(loss_history) * 100, 100), loss_history,
            color="purple", linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Gradient Descent — Loss Curve")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("linear_regression_results.png", dpi=150)
    print("\n  Plot saved as 'linear_regression_results.png'")
    plt.show()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print("=" * 50)
    print("   LINEAR REGRESSION — Educational Demo")
    print("=" * 50)

    # 1. Generate data
    print("\n[1] Generating synthetic dataset...")
    X, y = generate_dataset(n_samples=300, noise=25)

    # 2. Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"    Train samples : {X_train.shape[0]}")
    print(f"    Test  samples : {X_test.shape[0]}")

    # 3. Feature scaling (important for gradient descent)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 4. Train — from-scratch model
    print("\n[2] Training Linear Regression from scratch (Gradient Descent)...")
    scratch_model = LinearRegressionScratch(learning_rate=0.1, n_iterations=1000)
    scratch_model.fit(X_train_scaled, y_train)
    scratch_preds = scratch_model.predict(X_test_scaled)

    print(f"    Learned weight : {scratch_model.weights[0]:.4f}")
    print(f"    Learned bias   : {scratch_model.bias:.4f}")

    # 5. Train — scikit-learn model
    print("\n[3] Training Linear Regression with scikit-learn...")
    sklearn_model = SKLearnLinearRegression()
    sklearn_model.fit(X_train_scaled, y_train)
    sklearn_preds = sklearn_model.predict(X_test_scaled)

    print(f"    sklearn weight : {sklearn_model.coef_[0]:.4f}")
    print(f"    sklearn bias   : {sklearn_model.intercept_:.4f}")

    # 6. Evaluate both models
    print("\n[4] Evaluating models...")
    evaluate(y_test, scratch_preds, label="From-Scratch (Gradient Descent)")
    evaluate(y_test, sklearn_preds, label="scikit-learn (Closed-Form)")

    # 7. Visualize
    print("\n[5] Generating plots...")
    plot_all(
        X_train_scaled, X_test_scaled,
        y_train, y_test,
        scratch_preds, sklearn_preds,
        scratch_model.loss_history
    )

    print("\n✓ Done! Review the metrics above and the saved plot.")
    print("=" * 50)


if __name__ == "__main__":
    main()
