import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay

# 1. Load CSV
df = pd.read_csv("predictions_vs_true.csv")

# 2. Extract values
true_vals = df['True'].values
pred_vals = df['Predicted'].values

# 3. Compute metrics
rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
mae = mean_absolute_error(true_vals, pred_vals)

# Directional accuracy
true_diff = np.sign(np.diff(true_vals))
pred_diff = np.sign(np.diff(pred_vals))
directional_accuracy = np.sum(true_diff == pred_diff) / len(true_diff) * 100

# Accuracy within ±2 price units
within_tolerance = np.abs(true_vals - pred_vals) <= 3.0
tolerance_accuracy = np.mean(within_tolerance) * 100

# 4. Print results
print(f"Directional Accuracy: {directional_accuracy:.2f}%")
print(f"Accuracy (within ±3.0 price units): {tolerance_accuracy:.2f}%")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# 5. Plot actual vs predicted
plt.figure(figsize=(12, 6))
plt.plot(true_vals, label='True Close Price', linewidth=2)
plt.plot(pred_vals, label='Predicted Close Price', linestyle='--')
plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Time Step")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Plot error histogram
errors = pred_vals - true_vals
plt.figure(figsize=(8, 4))
plt.hist(errors, bins=50, color='skyblue', edgecolor='black')
plt.title("Prediction Error Histogram")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Confusion Matrix for direction prediction
cm = confusion_matrix(true_diff, pred_diff, labels=[-1, 0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Down', 'Flat', 'Up'])

plt.figure(figsize=(6, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix for Directional Prediction")
plt.grid(False)
plt.tight_layout()
plt.show()
