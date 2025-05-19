import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Fix random seeds
torch.manual_seed(42)
np.random.seed(42)

# 1) Download data
def download_data(ticker='AAPL', start='2015-01-01', end='2023-01-01'):
    print("[1] Downloading data...")
    data = yf.download(ticker, start=start, end=end)
    data.dropna(inplace=True)
    return data

# 2) Compute indicators: SMA, RSI, Volume scaling
def add_indicators(df):
    print("[2] Adding indicators (SMA, RSI, Volume)...")
    df = df.copy()
    df['SMA_14'] = df['Close'].rolling(window=14).mean()

    # RSI calculation
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # Normalize volume (min-max)
    df['Volume_Norm'] = (df['Volume'] - df['Volume'].min()) / (df['Volume'].max() - df['Volume'].min())

    df.dropna(inplace=True)
    return df

# 3) Prepare sequences
def create_sequences(features, close_scaled, seq_length=30):
    xs = []
    ys = []
    for i in range(len(features) - seq_length):
        x = features[i:i+seq_length]
        y = close_scaled[i+seq_length]
        xs.append(x)
        ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys).squeeze()
    return xs, ys

# 4) BiLSTM model
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

# 5) Training function
def train_model(model, X_train, y_train, epochs=300, lr=0.001, patience=15):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output.squeeze(), y_train)
        loss.backward()
        optimizer.step()

        scheduler.step(loss)

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            torch.save(model.state_dict(), 'model.pth')
        else:
            patience_counter += 1

        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}, Best Loss: {best_loss:.6f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# 6) Directional accuracy
def directional_accuracy(true_vals, preds):
    true_diff = np.diff(true_vals.squeeze())
    pred_diff = np.diff(preds.squeeze())
    correct = np.sum((true_diff * pred_diff) > 0)
    total = len(true_diff)
    return correct / total * 100

# 7) Accuracy with tolerance
def tolerance_accuracy(y_true, y_pred, tolerance=1.0):
    errors = np.abs(y_true - y_pred)
    correct = (errors < tolerance).astype(int)
    return correct.mean() * 100, errors, correct

def main():
    df = download_data()
    df = add_indicators(df)

    features = df[['SMA_14', 'RSI_14', 'Volume_Norm']].values
    close = df['Close'].values.reshape(-1, 1)

    scaler_feat = MinMaxScaler()
    scaler_close = MinMaxScaler()

    features_scaled = scaler_feat.fit_transform(features)
    close_scaled = scaler_close.fit_transform(close)

    seq_length = 30
    X, y = create_sequences(features_scaled, close_scaled, seq_length)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    model = BiLSTMModel(input_size=X.shape[2])
    train_model(model, X_tensor, y_tensor, epochs=300, lr=0.001, patience=15)

    print("Loading best model from model.pth for evaluation...")
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    with torch.no_grad():
        preds = model(X_tensor).squeeze().numpy()

    preds_inv = scaler_close.inverse_transform(preds.reshape(-1,1))
    y_inv = scaler_close.inverse_transform(y.reshape(-1,1))

    # Directional Accuracy
    acc_dir = directional_accuracy(y_inv, preds_inv)
    print(f"Directional Accuracy: {acc_dir:.2f}%")

    # Tolerance Accuracy
    acc_tol, errors, correct_flags = tolerance_accuracy(y_inv, preds_inv, tolerance=3.0)
    print(f"Accuracy (within ±3.0 price unit): {acc_tol:.2f}%")

    # RMSE and MAE
    rmse = np.sqrt(mean_squared_error(y_inv, preds_inv))
    mae = mean_absolute_error(y_inv, preds_inv)

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    # Plot results
    plt.figure(figsize=(12,6))
    plt.plot(y_inv, label='True Close Price')
    plt.plot(preds_inv, label='Predicted Close Price')
    plt.title("Stock Price Prediction using BiLSTM + SMA + RSI + Volume")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Show first 10 predictions
    print("First 10 predicted vs true close prices:")
    for i in range(10):
        print(f"Predicted: {preds_inv[i,0]:.2f}, True: {y_inv[i,0]:.2f}")

    # Save predictions to CSV
    results_df = pd.DataFrame({
        'Predicted': preds_inv.flatten(),
        'True': y_inv.flatten(),
        'Abs_Error': errors.flatten(),
        'Correct_within_tol': correct_flags.flatten()
    })
    results_df.to_csv("predictions_vs_true.csv", index=False)
    print("Results saved to 'predictions_vs_true.csv'.")

if __name__ == "__main__":
    main()
