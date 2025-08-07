from time import sleep
import uuid
import pandas as pd
from sklearn.metrics import mean_absolute_error
import streamlit as st
from streamlit_option_menu import option_menu
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from services import load_data, plot_data, plot_multiple_data, plot_volume

# Set page layout to wide
st.set_page_config(layout="wide", page_title="PredicTable", page_icon="📈")

# Sidebar
st.sidebar.markdown("<h1 style='text-align: center; font-size: 30px;'><b>Predic.</b><b style='color: orange'>Table</b></h1>", unsafe_allow_html=True)
st.sidebar.title("Options")
start_date_key = str(uuid.uuid4())
start_date = st.sidebar.date_input("Start date", date(2018, 1, 1), key=start_date_key)
end_date = st.sidebar.date_input("End date", date.today())

# Header
st.markdown("<h1 style='text-align: center;'>Stock Forecast App 📈</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'><b>Predic.</b><b style='color: orange'>Table</b> is a simple web app for stock price prediction using the <a href='https://facebook.github.io/prophet/'>Prophet</a> library.</p>", unsafe_allow_html=True)

selected_tab = option_menu(
    menu_title=None,
    options=["Dataframes", "Plots", "Statistics", "Forecasting", "Comparison"],
    icons=["table", "bar-chart", "calculator", "graph-up-arrow", "arrow-down-up"],
    menu_icon="📊",
    default_index=0,
    orientation="horizontal",
)

# Stock selection
stocks = ("AAPL", "GOOG", "MSFT", "GME", "AMC", "TSLA", "AMZN", "NFLX", "NVDA", "AMD", "PYPL")

# Stocks abreviations
selected_stock = st.sidebar.selectbox("Select stock for prediction", stocks)
selected_stocks = st.sidebar.multiselect("Select stocks for comparison", stocks)

years_to_predict = st.sidebar.slider("Years of prediction:", 1, 5)
period = years_to_predict * 365

# Display a loading spinner while loading data
with st.spinner("Loading data..."):
    data = load_data(selected_stock, start_date, end_date)
    sleep(1)

# Check if data was loaded successfully
if data is None or data.empty:
    st.error(f"Failed to load data for {selected_stock}. Please try again with different dates or stock symbol.")
    st.stop()

# Display the success message
success_message = st.success("Data loaded successfully!")

# Introduce a delay before clearing the success message
sleep(1)

# Clear the success message
success_message.empty()

# Forecasting
df_train = data[["Date", "Close"]].copy()
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Ensure the data types are correct
df_train['ds'] = pd.to_datetime(df_train['ds'])

# Reset index to ensure proper Series structure
df_train = df_train.reset_index(drop=True)

# Convert y column to numeric with robust handling
try:
    df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
except (TypeError, ValueError):
    # If direct conversion fails, extract values and convert
    y_values = df_train['y'].values if hasattr(df_train['y'], 'values') else df_train['y']
    # Flatten the array if it's 2D
    if hasattr(y_values, 'flatten'):
        y_values = y_values.flatten()
    df_train['y'] = pd.to_numeric(pd.Series(y_values), errors='coerce')

# Remove any rows with NaN values
df_train = df_train.dropna()

# Check if we have enough data points
if len(df_train) < 2:
    st.error("Not enough valid data points for forecasting. Please try different dates.")
    st.stop()

# Ensure proper DataFrame structure for Prophet
ds_values = df_train['ds'].values
y_values = df_train['y'].values

# Flatten arrays if they are 2D
if hasattr(ds_values, 'flatten') and ds_values.ndim > 1:
    ds_values = ds_values.flatten()
if hasattr(y_values, 'flatten') and y_values.ndim > 1:
    y_values = y_values.flatten()

df_train = pd.DataFrame({
    'ds': pd.to_datetime(ds_values),
    'y': pd.Series(y_values).astype(float)
})

model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

# Convert end_date to datetime
end_date_datetime = pd.to_datetime(end_date)

# Filter forecast based on end_date
forecast = forecast[forecast['ds'] >= end_date_datetime]

# Dataframes Tab
if selected_tab == "Dataframes":
    # Display historical data
    st.markdown("<h2><span style='color: orange;'>{}</span> Historical Data</h2>".format(selected_stock), unsafe_allow_html=True)
    st.write("This section displays historical stock price data for {} from {} to {}.".format(selected_stock, start_date, end_date))
    
    # Copy data
    new_data = data.copy()

    # Drop Adj Close and Volume columns if they exist
    columns_to_drop = ['Adj Close', 'Volume']
    columns_to_drop = [col for col in columns_to_drop if col in new_data.columns]
    if columns_to_drop:
        new_data = new_data.drop(columns=columns_to_drop)
    st.dataframe(new_data, use_container_width=True)

    # Display forecast data
    st.markdown("<h2><span style='color: orange;'>{}</span> Forecast Data</h2>".format(selected_stock), unsafe_allow_html=True)
    st.write("This section displays the forecasted stock price data for {} using the Prophet model from {} to {}.".format(selected_stock, end_date, end_date + pd.Timedelta(days=period)))
    
    # Copy forecast dataframe
    new_forecast = forecast.copy()

    # Drop unwanted columns
    new_forecast = new_forecast.drop(columns=[
        'additive_terms', 
        'additive_terms_lower', 
        'additive_terms_upper', 
        'weekly', 
        'weekly_lower', 
        'weekly_upper', 
        'yearly', 
        'yearly_lower', 
        'yearly_upper', 
        'multiplicative_terms', 
        'multiplicative_terms_lower', 
        'multiplicative_terms_upper'
    ])
    
    # Rename columns
    new_forecast = new_forecast.rename(columns={
        "ds": "Date", 
        "yhat": "Close", 
        "yhat_lower": "Close Lower",
        "yhat_upper": "Close Upper",
        "trend": "Trend", 
        "trend_lower": "Trend Lower", 
        "trend_upper": "Trend Upper"
    })

    st.dataframe(new_forecast, use_container_width=True)

# Plots Tab
if selected_tab == "Plots":
    # Raw data plot
    plot_data(data)

    # Data Volume plot
    plot_volume(data)

# Statistics Tab
if selected_tab == "Statistics":
    st.markdown("<h2><span style='color: orange;'>Descriptive </span>Statistics</h2>", unsafe_allow_html=True)
    st.write("This section provides descriptive statistics for the selected stock.")

    # Descriptive Statistics Table
    # drop the date column and other non-numeric columns if they exist
    stats_data = data.copy()
    columns_to_drop = ['Date', 'Adj Close', 'Volume']
    columns_to_drop = [col for col in columns_to_drop if col in stats_data.columns]
    if columns_to_drop:
        stats_data = stats_data.drop(columns=columns_to_drop)
    st.table(stats_data.describe())

# Forecasting Tab    
if selected_tab == "Forecasting":
    # Plotting forecast
    st.markdown("<h2><span style='color: orange;'>{}</span> Forecast Plot</h2>".format(selected_stock), unsafe_allow_html=True)
    st.write("This section visualizes the forecasted stock price for {} using a time series plot from {} to {}.".format(selected_stock, end_date, end_date + pd.Timedelta(days=period)))
    forecast_plot = plot_plotly(model, forecast)
    st.plotly_chart(forecast_plot, use_container_width=True)

    # Plotting forecast components
    st.markdown("<h2><span style='color: orange;'>{}</span> Forecast Components</h2>".format(selected_stock), unsafe_allow_html=True)
    st.write("This section breaks down the forecast components, including trends and seasonality, for {} from {} to {}.".format(selected_stock, end_date, end_date + pd.Timedelta(days=period)))
    components = model.plot_components(forecast)
    st.write(components)

# Comparison Tab
if selected_tab == "Comparison":
    if selected_stocks:
        # Forecast multiple stocks
        stocks_data = []
        forcasted_data = []
        for stock in selected_stocks:
            stocks_data.append(load_data(stock, start_date, end_date))

        st.markdown("<h2><span style='color: orange;'>{}</span> Forecast Comparison Plot</h2>".format(', '.join(selected_stocks)), unsafe_allow_html=True)
        st.write("This section visualizes the forecasted stock price for {} using a time series plot from {} to {}.".format(', '.join(selected_stocks), end_date, end_date + pd.Timedelta(days=period)))

        for i, data in enumerate(stocks_data):
            if data is not None and not data.empty:
                df_train = data[["Date", "Close"]].copy()
                df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
                
                # Ensure the data types are correct
                df_train['ds'] = pd.to_datetime(df_train['ds'])
                
                # Reset index to ensure proper Series structure
                df_train = df_train.reset_index(drop=True)
                
                # Convert y column to numeric with robust handling
                try:
                    df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
                except (TypeError, ValueError):
                    # If direct conversion fails, extract values and convert
                    y_values = df_train['y'].values if hasattr(df_train['y'], 'values') else df_train['y']
                    # Flatten the array if it's 2D
                    if hasattr(y_values, 'flatten'):
                        y_values = y_values.flatten()
                    df_train['y'] = pd.to_numeric(pd.Series(y_values), errors='coerce')
                
                # Remove any rows with NaN values
                df_train = df_train.dropna()
                
                # Check if we have enough data points
                if len(df_train) < 2:
                    st.warning(f"Not enough valid data points for {selected_stocks[i]}. Skipping this stock.")
                    continue
                
                # Ensure proper DataFrame structure for Prophet
                ds_values = df_train['ds'].values
                y_values = df_train['y'].values
                
                # Flatten arrays if they are 2D
                if hasattr(ds_values, 'flatten') and ds_values.ndim > 1:
                    ds_values = ds_values.flatten()
                if hasattr(y_values, 'flatten') and y_values.ndim > 1:
                    y_values = y_values.flatten()
                
                df_train = pd.DataFrame({
                    'ds': pd.to_datetime(ds_values),
                    'y': pd.Series(y_values).astype(float)
                })
                
                model = Prophet()
                model.fit(df_train)
                future = model.make_future_dataframe(periods=period)
                forecast = model.predict(future)
                forecast = forecast[forecast['ds'] >= end_date_datetime]
                st.markdown("<h3><span style='color: orange;'>{}</span> Forecast DataFrame</h3>".format(selected_stocks[i]), unsafe_allow_html=True)

                # Copy forecast dataframe
                new_forecast = forecast.copy()

                # Drop unwanted columns
                new_forecast = new_forecast.drop(columns=[
                    'additive_terms', 
                    'additive_terms_lower', 
                    'additive_terms_upper', 
                    'weekly', 
                    'weekly_lower', 
                    'weekly_upper', 
                    'yearly', 
                    'yearly_lower', 
                    'yearly_upper', 
                    'multiplicative_terms', 
                    'multiplicative_terms_lower', 
                    'multiplicative_terms_upper'
                ])

                # Rename columns
                new_forecast = new_forecast.rename(columns={
                    "ds": "Date", 
                    "yhat": "Close", 
                    "yhat_lower": "Close Lower",
                    "yhat_upper": "Close Upper",
                    "trend": "Trend", 
                    "trend_lower": "Trend Lower", 
                    "trend_upper": "Trend Upper"
                })

                st.dataframe(new_forecast, use_container_width=True)

                forcasted_data.append(forecast)

        plot_multiple_data(forcasted_data, selected_stocks)
    else:
        st.warning("Please select at least one stock if you want to compare them.")

# Display balloons at the end
# st.balloons()