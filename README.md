
# LSTM Time Series Forecaster 🚀

Welcome to **LSTM Time Series Forecaster**, a flexible Python-based forecasting tool that uses Long Short-Term Memory (LSTM) neural networks. This script, powered by PyCaret, simplifies the process of loading, preparing, training, and evaluating LSTM models for a variety of time series data.

Created and maintained by [mr-veyrion](https://github.com/mr-veyrion).

---

## ✨ Features

- **📚 Multi-Source Data Handling:**
  - Built-in PyCaret datasets (`airline`, `uschange`, etc.).
  - Local CSV files.
  - Yahoo Finance ticker data (e.g., AAPL, GOOGL).

- **🧠 Custom LSTM Architecture:**
  - Adjustable sequence length (look-back window).
  - Configurable LSTM units, activation, optimizer, and loss function.
  - Control over epochs and batch size.

- **📈 Flexible Forecasting:**
  - Define any custom forecast horizon.
  - Visualise predictions versus actuals.
  - Output plots and CSVs for further analysis.

- **⚙️ Integrated with PyCaret:**
  - Efficient time series setup.
  - Smart train-test splits based on forecast horizon.

---

## 📋 Prerequisites

- Python 3.7 to 3.10
- Conda (recommended) or virtualenv

---

## 🛠️ Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/mr-veyrion/timeseries-lstm-forecast.git
   cd timeseries-lstm-forecast

2. **Create and Activate Environment**

   ```bash
   conda create -n tsforecast python=3.9 -y
   conda activate tsforecast

   *Or using `venv`:*

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Unix or MacOS
   venv\Scripts\activate      # On Windows
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *Note: `tensorflow` in `requirements.txt` is CPU-based. For GPU, install `tensorflow-gpu` manually and configure CUDA/cuDNN.*

---

## 🚀 Quick Start (Airline Dataset)

Run the script using default parameters:

```bash
python forecast.py
```

This will:

* Load the `airline` dataset.
* Forecast 12 future periods using past 12 months.
* Display a forecast plot and print the MAE (Mean Absolute Error).

---

## ⚙️ Custom Usage

The heart of the project is `forecast.py`. You can control its behaviour with CLI arguments:

```bash
python forecast.py [OPTIONS]
```

### 🔧 Common Options:

**Data Config:**

* `--data_source` → `pycaret`, `csv`, or `yfinance` (default: `pycaret`)
* `--dataset_name` → PyCaret dataset or Yahoo ticker (e.g., `airline`, `GOOGL`)
* `--csv_path` → Path to your CSV file (if `data_source=csv`)
* `--date_column` → Column name for dates (default: `Date`)
* `--value_column` → Column name for values (default: `Value`)
* `--yf_value_column` → Yahoo Finance field (e.g., `Close`, `Adj Close`)

**Forecasting & Model:**

* `--forecast_horizon` or `-fh` → Future steps to predict (default: `12`)
* `--seq_length` → Look-back window (default: `12`)
* `--lstm_units` → LSTM layer units (default: `50`)
* `--epochs` → Training epochs (default: `100`)
* `--batch_size` → Training batch size (default: `32`)

**Output Control:**

* `--plot_title` → Title for the plot
* `--output_plot_path` → Path to save the plot
* `--output_predictions_path` → Path to save predictions (CSV)

Run `python forecast.py --help` to explore all options.

---

## 📌 Examples

### Forecast a PyCaret dataset:

```bash
python forecast.py --data_source pycaret --dataset_name uschange --fh 24 --seq_length 12 --epochs 150
```

### Forecast from CSV:

```bash
python forecast.py \
    --data_source csv \
    --csv_path path/to/my_data.csv \
    --date_column TransactionDate \
    --value_column SalesAmount \
    --fh 6 \
    --seq_length 10 \
    --output_plot_path output/sales_forecast.png \
    --output_predictions_path output/sales_predictions.csv
```

### Forecast Google stock prices:

```bash
python forecast.py \
    --data_source yfinance \
    --dataset_name GOOGL \
    --yf_value_column Close \
    --fh 30 \
    --seq_length 60 \
    --epochs 200 \
    --plot_title "GOOGL Stock Price Forecast (Next 30 Days)"
```

---

## 📊 Outputs

* **Console:** Logs training progress and MAE.
* **Forecast Plot:** Shows:

  * Historical training data.
  * Actual future values.
  * LSTM predictions.
* **Predictions CSV:** If specified, outputs actual and predicted values with timestamps.

---

## 🧠 Author

**Abhishek Sharma** ([mr-veyrion](https://github.com/mr-veyrion))
AI Engineer | Time Series | LSTM
