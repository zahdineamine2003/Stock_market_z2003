
# Stock Market Price Prediction using BiLSTM with Technical Indicators

This project predicts stock closing prices using a Bidirectional LSTM (BiLSTM) deep learning model enhanced with technical indicators such as SMA, RSI, and normalized volume.

---

## Features

- Download and preprocess stock data from Yahoo Finance
- Compute technical indicators:  
  - **SMA** (Simple Moving Average)  
  - **RSI** (Relative Strength Index)  
  - Normalized Volume  
- Prepare sequences of features for LSTM input
- Train BiLSTM model to predict next-day closing prices
- Early stopping and learning rate scheduling for optimized training
- Evaluate predictions with:  
  - Directional Accuracy  
  - RMSE (Root Mean Squared Error)  
  - MAE (Mean Absolute Error)  
  - Accuracy within tolerance (price unit)
- Save trained model weights (`model.pth`, `best_model.pth`)
- Save prediction results in CSV files (e.g., `predictions_vs_true.csv`)
- Visualization and evaluation through `evaluate_predictions.py`  
- Support for multiple stocks with example monthly prediction CSVs (`AAPL_monthly_predicted_prices.csv`, `MSFT_monthly_predicted_prices.csv`, `META_monthly_predicted_prices.csv`)

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/zahdineamine2003/Stock_market_z2003.git
   cd Stock_market_z2003
````

2. (Optional) Create and activate a virtual environment:

   * On Linux/macOS:

     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   * On Windows:

     ```powershell
     python -m venv venv
     venv\Scripts\activate
     ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Train and Predict Stock Prices

Run the main training and prediction script:

```bash
python predict_stocks.py
```

This script will:

* Download stock data (default is AAPL from 2015 to 2023)
* Add SMA, RSI, and normalized volume indicators
* Prepare data sequences for BiLSTM input
* Train the BiLSTM model with early stopping
* Save the best model weights (`model.pth`, `best_model.pth`)
* Save predictions and true values in CSV files (e.g., `predictions_vs_true.csv`)

### Evaluate Predictions

To visualize and analyze model predictions including metrics and confusion matrix, run:

```bash
python evaluate_predictions.py
```

This will:

* Load the prediction CSV file
* Calculate and print metrics: Directional Accuracy, RMSE, MAE, and accuracy with ±2 price units tolerance
* Plot true vs predicted prices
* Display a confusion matrix based on price movement direction

---

## Project Structure

```
.
│   AAPL_monthly_predicted_prices.csv
│   best_model.pth
│   evaluate_predictions.py
│   LICENSE
│   META_monthly_predicted_prices.csv
│   model.pth
│   MSFT_monthly_predicted_prices.csv
│   predictions_vs_true.csv
│   predict_stocks.py
│   requirements.txt
└───__pycache__
        train.cpython-313.pyc
```

* `predict_stocks.py`: Script to train the BiLSTM model and generate predictions
* `evaluate_predictions.py`: Script to evaluate predictions and plot graphs
* `model.pth`, `best_model.pth`: Saved model weights
* `predictions_vs_true.csv`, etc.: CSV files containing predictions and true values for analysis
* `requirements.txt`: Python dependencies
* `LICENSE`: License file

---

## Metrics

* **Directional Accuracy**: Percentage of times the predicted price movement direction matches the true direction.
* **RMSE** (Root Mean Squared Error): Measures average magnitude of prediction error.
* **MAE** (Mean Absolute Error): Measures average absolute error between predictions and true values.
* **Accuracy (within ±2 price units)**: Percentage of predictions within ±2 units of the true price.

---

## License

This project is licensed under the MIT License.

---

## Contact

Developed by Amine Zahdine
GitHub: [zahdineamine2003](https://github.com/zahdineamine2003)

```

---

```
