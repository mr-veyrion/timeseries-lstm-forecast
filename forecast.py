
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from pycaret.time_series import setup, get_config
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from pycaret.datasets import get_data as get_pycaret_data
import yfinance as yf
import argparse
import os

# --- Helper Function for LSTM Dataset Creation ---
def create_dataset(dataset, seq_length, forecast_horizon):
    X, y = [], []
    if len(dataset) < seq_length + forecast_horizon:
        return np.array(X), np.array(y) # Not enough data

    for i in range(len(dataset) - seq_length - forecast_horizon + 1):
        a = dataset[i:(i + seq_length), 0]
        X.append(a)
        target = dataset[i + seq_length : i + seq_length + forecast_horizon, 0]
        y.append(target)
    return np.array(X), np.array(y)

# --- Main Forecasting Function ---
def run_forecast(args):
    print("--- Starting Time Series Forecast ---")

    # 1. Load Data
    data = None
    target_column = args.value_column # Default for CSV

    if args.data_source == 'pycaret':
        try:
            print(f"Loading PyCaret dataset: {args.dataset_name}")
            data = get_pycaret_data(args.dataset_name)
            if isinstance(data.index, pd.PeriodIndex):
                data.index = data.index.to_timestamp()
            # For pycaret datasets, usually the column is the dataset name or a known one
            # If it's a series, convert to frame and name the column
            if isinstance(data, pd.Series):
                data = data.to_frame(name='Value') # Use a generic name
            target_column = data.columns[0] # Assume first column is target if not specified
            data.index.name = 'Date' # Ensure index has a name
            data = data.reset_index() # Make 'Date' a column for PyCaret setup if needed
            args.date_column = 'Date' # For PyCaret target and index setting

        except Exception as e:
            print(f"Error loading PyCaret dataset '{args.dataset_name}': {e}")
            return
    elif args.data_source == 'csv':
        if not args.csv_path:
            print("Error: CSV path must be provided for 'csv' data source.")
            return
        try:
            print(f"Loading CSV dataset from: {args.csv_path}")
            data = pd.read_csv(args.csv_path)
            if args.date_column not in data.columns:
                print(f"Error: Date column '{args.date_column}' not found in CSV.")
                return
            if args.value_column not in data.columns:
                print(f"Error: Value column '{args.value_column}' not found in CSV.")
                return
            data[args.date_column] = pd.to_datetime(data[args.date_column])
            data = data.set_index(args.date_column)
            data = data[[args.value_column]] # Keep only the value column
            target_column = args.value_column
        except Exception as e:
            print(f"Error loading CSV file '{args.csv_path}': {e}")
            return
    elif args.data_source == 'yfinance':
        if not args.dataset_name:
            print("Error: Ticker symbol (dataset_name) must be provided for 'yfinance' source.")
            return
        try:
            print(f"Fetching data for ticker: {args.dataset_name} from Yahoo Finance")
            data = yf.download(args.dataset_name, start=args.yf_start_date, end=args.yf_end_date)
            if data.empty:
                print(f"No data found for ticker {args.dataset_name}")
                return
            data = data[[args.yf_value_column]] # e.g., 'Close' or 'Adj Close'
            data.columns = ['Value'] # Rename to generic 'Value'
            target_column = 'Value'
            # PyCaret needs the date as a column if it's not a PeriodIndex
            data.index.name = 'Date'
            data = data.reset_index()
            args.date_column = 'Date'
            args.value_column = 'Value'

        except Exception as e:
            print(f"Error fetching data from Yahoo Finance for '{args.dataset_name}': {e}")
            return
    else:
        print(f"Error: Unknown data source '{args.data_source}'")
        return

    if data is None or data.empty:
        print("No data loaded. Exiting.")
        return

    print(f"Data loaded successfully. Shape: {data.shape}")
    print(f"Target column for forecasting: {target_column}")
    print(f"Date column for PyCaret setup: {args.date_column}")


    # 2. Set up PyCaret environment
    print(f"\nSetting up PyCaret environment. Forecast Horizon (fh): {args.forecast_horizon}")
    # Ensure the target data for PyCaret is a Series with a DatetimeIndex or PeriodIndex
    # If data came from CSV or yfinance and index was reset:
    if args.date_column in data.columns and target_column in data.columns:
        pycaret_data_series = data.set_index(args.date_column)[target_column]
    else: # Assumes data is already a Series or DataFrame with index as date
        pycaret_data_series = data[target_column] if isinstance(data, pd.DataFrame) else data

    if not isinstance(pycaret_data_series.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        print("Error: Data index must be DatetimeIndex or PeriodIndex for PyCaret setup.")
        print(f"Current index type: {type(pycaret_data_series.index)}")
        return

    # Ensure the data passed to setup is sorted by index
    pycaret_data_series = pycaret_data_series.sort_index()

    # Check if enough data for fh
    if len(pycaret_data_series) <= args.forecast_horizon:
        print(f"Error: Not enough data ({len(pycaret_data_series)} points) for the forecast horizon ({args.forecast_horizon}).")
        print("Please use a smaller forecast horizon or provide more data.")
        return

    try:
        setup(data=pycaret_data_series, fh=args.forecast_horizon, fold=args.fold, session_id=args.session_id, verbose=False)
    except Exception as e:
        print(f"Error during PyCaret setup: {e}")
        print("This might be due to data format issues (e.g., non-numeric target, insufficient data length for fh).")
        print(f"Data head for PyCaret:\n{pycaret_data_series.head()}")
        return

    # 3. Get training and testing data from PyCaret
    train_data_series = get_config('y_train')
    test_data_series = get_config('y_test')

    train_data_values = train_data_series.values.reshape(-1, 1)
    test_data_values = test_data_series.values.reshape(-1, 1)

    print(f"Training data shape: {train_data_values.shape}")
    print(f"Test data shape: {test_data_values.shape}")

    # 4. Create dataset for LSTM
    # seq_length for LSTM should be based on available training data
    # Ensure seq_length is not too large for the training data size
    actual_seq_length = min(args.seq_length, len(train_data_values) - args.forecast_horizon -1)
    if actual_seq_length < 1:
         print(f"Error: Not enough training data to form sequences with seq_length={args.seq_length} and fh={args.forecast_horizon}.")
         print(f"Training data length: {len(train_data_values)}. Need at least {args.seq_length + args.forecast_horizon} points.")
         return
    if actual_seq_length != args.seq_length:
        print(f"Warning: sequence length adjusted from {args.seq_length} to {actual_seq_length} due to training data size.")


    print(f"\nCreating LSTM sequences. Sequence Length: {actual_seq_length}, Forecast Horizon: {args.forecast_horizon}")
    X_train, y_train = create_dataset(train_data_values, actual_seq_length, args.forecast_horizon)

    if X_train.shape[0] == 0:
        print("Error: Not enough data in training set to create LSTM sequences.")
        print(f"  Need at least {actual_seq_length + args.forecast_horizon} data points in y_train.")
        print(f"  Current y_train length: {len(train_data_values)}")
        return
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # 5. Define and compile the LSTM model
    print("\nDefining and compiling LSTM model...")
    model = Sequential([
        LSTM(args.lstm_units, activation=args.lstm_activation, input_shape=(actual_seq_length, 1)),
        Dense(args.forecast_horizon) # Output layer predicts all steps in the horizon
    ])
    model.compile(optimizer=args.optimizer, loss=args.loss, metrics=['mae'])
    model.summary()

    # 6. Train the model
    print("\nTraining LSTM model...")
    model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=args.verbose)

    # 7. Generate predictions
    print("\nGenerating predictions...")
    # Prepare input sequence from the last seq_length points of *actual* training data
    if len(train_data_values) < actual_seq_length:
        print(f"Error: Not enough data in the original train_data ({len(train_data_values)} points) to form the last sequence of length {actual_seq_length}.")
        return

    last_seq_input = train_data_values[-actual_seq_length:].reshape((1, actual_seq_length, 1))
    predictions = model.predict(last_seq_input).flatten()

    # 8. Evaluate
    # Ensure test_data_values is trimmed/padded to match forecast_horizon if PyCaret fh differs
    # PyCaret's y_test should already match args.forecast_horizon due to setup(fh=...)
    test_data_trimmed_for_eval = test_data_values[:args.forecast_horizon].flatten()

    if len(test_data_trimmed_for_eval) != len(predictions):
        print(f"Warning: Length mismatch between test data ({len(test_data_trimmed_for_eval)}) and predictions ({len(predictions)}). This should not happen if fh is consistent.")
        # This might occur if fh in PyCaret setup was different or test set is smaller than fh
        min_len = min(len(test_data_trimmed_for_eval), len(predictions))
        test_data_trimmed_for_eval = test_data_trimmed_for_eval[:min_len]
        predictions = predictions[:min_len]
        print(f"Adjusted to compare {min_len} points.")


    mae = mean_absolute_error(test_data_trimmed_for_eval, predictions)
    print(f"\n--- Evaluation ---")
    print(f"Mean Absolute Error (MAE) on test set: {mae:.4f}")

    # 9. Plot results
    print("\nPlotting results...")
    plt.figure(figsize=(12, 7))

    # Full historical data for context (using original pycaret_data_series index)
    full_series_index = pycaret_data_series.index

    # Training data plot
    train_indices = full_series_index[:len(train_data_series)]
    plt.plot(train_indices, train_data_series.values, label='Training Data')

    # Test data plot
    test_indices = full_series_index[len(train_data_series) : len(train_data_series) + len(test_data_trimmed_for_eval)]
    plt.plot(test_indices, test_data_trimmed_for_eval, label='Actual Test Data', marker='o')

    # Predictions plot (should align with test_indices)
    plt.plot(test_indices, predictions, label='LSTM Predictions', linestyle='--', marker='x')

    plt.title(f'{args.plot_title} (Seq: {actual_seq_length}, FH: {args.forecast_horizon}, LSTM Units: {args.lstm_units})')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if args.output_plot_path:
        if not os.path.exists(os.path.dirname(args.output_plot_path)) and os.path.dirname(args.output_plot_path) != '':
            os.makedirs(os.path.dirname(args.output_plot_path), exist_ok=True)
        plt.savefig(args.output_plot_path)
        print(f"Plot saved to {args.output_plot_path}")
    plt.show()

    # Save predictions if path provided
    if args.output_predictions_path:
        pred_df = pd.DataFrame({'Date': test_indices, 'Actual': test_data_trimmed_for_eval, 'Predicted': predictions})
        if not os.path.exists(os.path.dirname(args.output_predictions_path)) and os.path.dirname(args.output_predictions_path) != '':
            os.makedirs(os.path.dirname(args.output_predictions_path), exist_ok=True)
        pred_df.to_csv(args.output_predictions_path, index=False)
        print(f"Predictions saved to {args.output_predictions_path}")

    print("\n--- Forecast Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Series Forecasting using LSTM with PyCaret.")

    # Data Source Arguments
    parser.add_argument('--data_source', type=str, default='pycaret', choices=['pycaret', 'csv', 'yfinance'],
                        help="Source of the data ('pycaret', 'csv', 'yfinance').")
    parser.add_argument('--dataset_name', type=str, default='airline',
                        help="Name of the dataset (e.g., 'airline' for PyCaret, ticker for yfinance like 'AAPL').")
    parser.add_argument('--csv_path', type=str, help="Path to the CSV file if data_source is 'csv'.")
    parser.add_argument('--date_column', type=str, default='Date', help="Name of the date column in CSV.")
    parser.add_argument('--value_column', type=str, default='Value', help="Name of the value column in CSV / target for PyCaret.")

    # Yahoo Finance specific arguments
    parser.add_argument('--yf_start_date', type=str, default='2010-01-01', help="Start date for yfinance data (YYYY-MM-DD).")
    parser.add_argument('--yf_end_date', type=str, default=pd.Timestamp.today().strftime('%Y-%m-%d'), help="End date for yfinance data (YYYY-MM-DD).")
    parser.add_argument('--yf_value_column', type=str, default='Close', help="Column to use from yfinance data (e.g., 'Close', 'Adj Close').")

    # PyCaret Setup Arguments
    parser.add_argument('--forecast_horizon', '-fh', type=int, default=12, help="Forecast horizon (number of steps to predict).")
    parser.add_argument('--fold', type=int, default=3, help="Number of folds for PyCaret cross-validation setup.")
    parser.add_argument('--session_id', type=int, default=42, help="Random seed for reproducibility.")

    # LSTM Model Arguments
    parser.add_argument('--seq_length', type=int, default=12,
                        help="Length of input sequences for LSTM.")
    parser.add_argument('--lstm_units', type=int, default=50, help="Number of units in the LSTM layer.")
    parser.add_argument('--lstm_activation', type=str, default='relu', help="Activation function for LSTM layer.")
    parser.add_argument('--optimizer', type=str, default='adam', help="Optimizer for model training.")
    parser.add_argument('--loss', type=str, default='mse', help="Loss function for model training.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2], help="Verbosity mode for model training (0=silent, 1=progress bar, 2=one line per epoch).")

    # Output Arguments
    parser.add_argument('--plot_title', type=str, default='Time Series Forecast', help="Title for the output plot.")
    parser.add_argument('--output_plot_path', type=str, help="Path to save the output plot (e.g., output/forecast.png).")
    parser.add_argument('--output_predictions_path', type=str, help="Path to save predictions to a CSV file (e.g., output/predictions.csv).")


    args = parser.parse_args()

    # Create output directory if specified and doesn't exist
    if args.output_plot_path and not os.path.exists(os.path.dirname(args.output_plot_path)) and os.path.dirname(args.output_plot_path) != '':
        os.makedirs(os.path.dirname(args.output_plot_path), exist_ok=True)
    if args.output_predictions_path and not os.path.exists(os.path.dirname(args.output_predictions_path)) and os.path.dirname(args.output_predictions_path) != '':
        os.makedirs(os.path.dirname(args.output_predictions_path), exist_ok=True)

    run_forecast(args)
