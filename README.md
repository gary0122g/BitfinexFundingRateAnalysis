# Bitfinex fUSD Funding Rate Prediction System

## Project Overview

This project is a prediction system for Bitfinex fUSD funding rates based on LSTM deep learning models. It collects historical fUSD funding rate data and BTC price data from the Bitfinex exchange to build a multi-feature LSTM model that predicts future funding rate trends.[1]

## Features

- Automatic retrieval of fUSD funding rate and BTC price historical data from Bitfinex API
- Data visualization analysis, including funding rate trends and BTC price comparisons
- Multi-feature prediction system based on LSTM deep learning models
- Feature correlation analysis to evaluate the impact of BTC price on fUSD funding rates
- Prediction of fUSD funding rate trends for the next 7 days
- Model performance evaluation and comparison (models with and without BTC features)[3]

## Installation Requirements

```
pip install requests pandas numpy matplotlib scikit-learn tensorflow
```

## Main Modules

### 1. Data Acquisition

- `get_fusd_funding_stats()`: Retrieves Bitfinex fUSD funding statistics data
- `get_btc_price_data()`: Retrieves Bitfinex BTC/USD price data
- `get_historical_fusd_data()`: Batch retrieval of historical fUSD funding rate data[5]

### 2. Data Processing and Visualization

- `merge_fusd_and_btc_data()`: Merges fUSD and BTC data
- `plot_funding_rates()`: Plots funding rate trend charts
- `plot_feature_importance()`: Analyzes and plots feature correlations[7]

### 3. LSTM Model Building and Training

- `create_multifeature_sequences()`: Creates multi-feature sequence data
- `build_multifeature_lstm_model()`: Builds multi-feature LSTM model
- `train_multifeature_lstm_model()`: Trains multi-feature LSTM model[7]

### 4. Model Evaluation and Prediction

- `evaluate_multifeature_model()`: Evaluates model performance
- `plot_multifeature_predictions()`: Plots prediction results and forecasts future trends[2]

## Usage

Simply run the main function to start the complete process:

```python
if __name__ == "__main__":
    main()
```

The program will automatically execute the following steps:
1. Retrieve the latest fUSD funding rate
2. Retrieve historical fUSD funding rate data (default 90 days)
3. Retrieve BTC price data for the corresponding time period
4. Merge fUSD and BTC data
5. Plot funding rate trend charts and BTC price comparison charts
6. Analyze feature correlations
7. Train multi-feature LSTM model
8. Evaluate model performance
9. Predict funding rate trends for the next 7 days
10. Compare performance differences between models with and without BTC features[6]

## Output Files

The program generates the following files after execution:
- `bitfinex_fusd_btc_merged_data_YYYYMMDD.csv`: Merged fUSD and BTC data
- `fusd_funding_rate.png`: fUSD funding rate trend chart
- `btc_fusd_comparison.png`: BTC price and fUSD funding rate comparison chart
- `feature_correlation.png`: Feature correlation heatmap
- `target_correlation.png`: Correlation bar chart with target variable
- `multifeature_lstm_training_history.png`: Model training history
- `fusd_multifeature_prediction.png`: Prediction results and future trend chart
- `fusd_btc_predictions_YYYYMMDD.csv`: Prediction results data
- `fusd_btc_lstm_model_YYYYMMDD.h5`: Saved model file[7]

## Model Architecture

The LSTM model architecture consists of:
- Input layer with shape (sequence_length, n_features)
- First LSTM layer with return sequences enabled
- Dropout layer (0.5)
- Second LSTM layer
- Dropout layer (0.2)
- Dense output layer

The model is compiled with Adam optimizer and mean squared error loss function.[7]

## Performance Metrics

The model performance is evaluated using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Coefficient of Determination (R²)

The system also compares models with and without BTC price features to quantify the impact of BTC price on prediction accuracy.[2]

## Example Results

When running the program, you'll see outputs like:

```
Current fUSD daily rate: 0.0123%
Current fUSD annualized rate (APR): 4.4895%

Future fUSD Funding Rate Predictions (Multi-feature Model):
2025-04-26: 0.0125%
2025-04-27: 0.0127%
2025-04-28: 0.0129%
2025-04-29: 0.0130%
2025-04-30: 0.0128%
2025-05-01: 0.0126%
2025-05-02: 0.0124%

Model Performance Comparison:
Model with BTC features R²: 0.876543
Model without BTC features R²: 0.765432
Adding BTC features improved model performance by 14.52%
```

## Notes

- The API has rate limits, so the code includes sleep intervals between requests
- The model assumes a relationship between BTC price movements and fUSD funding rates
- Prediction accuracy depends on the quality and quantity of historical data
- The default sequence length is 8, meaning the model uses 8 previous data points to predict the next point[6]

## References

This project uses the Bitfinex API for data collection. For more information about Bitfinex's funding rates and API documentation, visit the [Bitfinex Documentation](https://docs.bitfinex.com/).[1]