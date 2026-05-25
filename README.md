# syn-intern

## Linear Regression — Educational Demo

A beginner-friendly implementation of Linear Regression built **from scratch** using NumPy, alongside a comparison with **scikit-learn**.

### Concepts Covered
- What is Linear Regression and the hypothesis function
- Cost Function (Mean Squared Error)
- Gradient Descent optimization step-by-step
- Feature scaling with StandardScaler
- Model evaluation: MSE, RMSE, MAE, R²
- Visualization of data, predictions, and training loss curve

### Files
| File | Description |
|------|-------------|
| `linear_regression.py` | Main program |
| `requirements.txt` | Python dependencies |
| `linear_regression_results.png` | Output plot (generated on run) |

### Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the program
python linear_regression.py
```

### What to Expect
```
==================================================
   LINEAR REGRESSION — Educational Demo
==================================================
[1] Generating synthetic dataset...
[2] Training from scratch (Gradient Descent)...
[3] Training with scikit-learn...
[4] Evaluating models...
[5] Generating plots...
```
A PNG plot is saved showing the dataset, predictions vs actual values, and the gradient descent loss curve.
