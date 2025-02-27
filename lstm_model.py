import bcrypt
import validators
import requests
import time
import tempfile

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

from tensorflow.keras.layers import Input, Dropout, LSTM, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping

from yahooquery import search
from io import StringIO
from fpdf import FPDF
from io import BytesIO
import base64

# Initialize session state for login
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def check_login(username, password):
    """Verify username and password against stored credentials."""
    users = st.secrets["credentials"]["users"]
    hashed_passwords = st.secrets["credentials"]["passwords"]

    if username in users:
        index = users.index(username)
        return bcrypt.checkpw(password.encode(), hashed_passwords[index].encode())
    return False

# If not authenticated, show login form
if not st.session_state.authenticated:
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if check_login(username, password):
            st.session_state.authenticated = True
            st.success("Login successful!")
            st.rerun()  # Refresh the page
        else:
            st.error("Invalid username or password")
    st.stop() # Prevents further execution until logged in

# If authenticated, show main content
if st.session_state.authenticated:
    st.sidebar.button("Logout", on_click=lambda: st.session_state.update(authenticated=False))

@st.cache_data
def get_stock_data_securely(ticker, timeout=10, max_retries=3):
    """Retrieves stock data."""
    attempt = 0
    while attempt < max_retries:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="max")
            if data.empty:
                raise ValueError("No data returned. Ticker may be incorrect or unavailable.")
            return data
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            attempt += 1
            if attempt == max_retries:
                raise Exception("Yahoo Finance API is unreachable. Please try again later.")
            time.sleep(2)
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")

def is_valid_url(url):
    """Checks if URL is valid and points to a CSV/XLSX file."""
    if not validators.url(url):
        return False
    if not (url.endswith(".csv") or url.endswith(".xlsx")):
        return False
    return True

def get_closing_price(data, date):
    """Finds the closing price from a date"""
    date = pd.to_datetime(date)
    return data.loc[date, 'Close'] if date in data.index else "Date not in range"

def get_next_available_date(selected_date, data):
    """Find the next available date in the dataset."""
    if selected_date > data.index.max():
        st.warning(f"The selected date {selected_date.date()} is in the future. Please choose a valid date within the available range.")
        return data.index.max()
    if selected_date < data.index.min():
        st.warning(f"The selected date {selected_date.date()} is too far in the past. Please choose a valid date within the available range.")
        return data.index.min()
    st.info("Stock market may be closed on selected date. Finding closing price of next available trading day.")
    next_available_date = data.index[data.index.searchsorted(selected_date)]
    return next_available_date

def detect_close_column(data):
    """Checks some common column names and convert them to 'Close.'"""
    possible_names = ["Close", "Close/Last", "Closing Price"]
    for col in data.columns:
        if any(col.strip().lower() == name.lower() for name in possible_names):
            return col
    return None

def create_sequences(data, seq_length):
    """Create sequences for LSTM Model."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    # return np.array(X), np.array(y).reshape(-1,1)
    return np.array(X), np.array(y)

from fpdf import FPDF

# Function to generate the PDF report
def generate_pdf(summary_stats, performance_metrics_option, selected_plot_keys, plots, model_plots, performance_metrics=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add Summary Statistics Table
    if summary_stats:
        pdf.cell(200, 10, txt="Summary Statistics", ln=True, align='C')
        stats = data.describe().reset_index()
        for i in range(len(stats)):
            row = stats.iloc[i]
            pdf.cell(200, 10, txt=f"{row['index']}: {row['Close']:.4f}", ln=True, align='L')

    # Add Performance Metrics
    if performance_metrics_option and performance_metrics:
        pdf.add_page()
        pdf.cell(200, 10, txt="Performance Metrics", ln=True, align='C')
        for metric, value in performance_metrics.items():
            pdf.cell(200, 10, txt=f"{metric}: {value:.4f}", ln=True, align='L')

    # Add Plots from both dictionaries
    for plot_key in selected_plot_keys:
        if plot_key in plots:
            fig = plots[plot_key]
        elif plot_key in model_plots:
            fig = model_plots[plot_key]
        else:
            continue
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            fig.savefig(tmpfile.name)
            pdf.add_page()
            pdf.cell(200, 10, txt=plot_key, ln=True, align='C')
            pdf.image(tmpfile.name, x=10, y=20, w=180)

    return pdf

# Function to generate HTML report
def generate_html(summary_stats, performance_metrics_option, selected_plot_keys, plots, model_plots, performance_metrics=None):
    html = StringIO()
    html.write("<html><head><title>Report</title></head><body>")
    html.write("<h1>Report</h1>")

    # Add Summary Statistics Table
    if summary_stats:
        html.write("<h2>Summary Statistics</h2>")
        stats = data.describe().reset_index()
        html.write(stats.to_html())

    # Add Performance Metrics
    if performance_metrics_option and performance_metrics:
        html.write("<h2>Performance Metrics</h2>")
        for metric, value in performance_metrics.items():
            html.write(f"<p>{metric}: {value:.4f}</p>")

    # Add Plots from both dictionaries
    for plot_key in selected_plot_keys:
        if plot_key in plots:
            fig = plots[plot_key]
        elif plot_key in model_plots:
            fig = model_plots[plot_key]
        else:
            continue
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            fig.savefig(tmpfile.name)
            html.write(f"<h2>{plot_key}</h2>")
            html.write(f'<img src="data:image/png;base64,{base64.b64encode(open(tmpfile.name, "rb").read()).decode()}" width="600"/>')

    html.write("</body></html>")
    return html.getvalue()

st.title("Stock Price LSTM Model")

# Set file size limit (in bytes)
FILE_SIZE_LIMIT = 30 * 1024 * 1024  # 30 MB limit

data_source = st.radio("Choose Data Source:", ("Yahoo Finance", "Upload File", "Enter URL"))
data = None

# To retrieve data with Yahoo Finance
if data_source == "Yahoo Finance":
    user_input = st.text_input("Enter Company Name or Stock Ticker:")
    if user_input:
        try:
            search_result = search(user_input)
            if search_result and "quotes" in search_result and search_result["quotes"]:
                ticker = search_result["quotes"][0]["symbol"].upper()
                st.write(f"Identified Ticker: {ticker}")
            else:
                st.error("No matching ticker found. Try another name.")
                st.stop()
        except Exception:
            st.error("Error retrieving ticker. Please try again.")
            st.stop()
        try:
            data = get_stock_data_securely(ticker)
            data.index = pd.to_datetime(data.index).date
        except Exception:
            st.error("Error retrieving stock data. Please try again.")
            st.stop()

# To retrieve data by upload or URL input
elif data_source in ["Upload File", "Enter URL"]:
    uploaded_file = st.file_uploader("Upload File", type=["csv", "xlsx"], accept_multiple_files=False) if data_source == "Upload File" else None
    url_input = st.text_input("Enter CSV URL:") if data_source == "Enter URL" else None
    if uploaded_file:
        if uploaded_file.size > FILE_SIZE_LIMIT:
            st.error(f"File size exceeds limit of {FILE_SIZE_LIMIT / (1024 * 1024)} MB. Please upload a smaller file. The limit is not 200MB per file.")
            st.stop()
        else:
            try:
                data = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col="Date")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                st.stop()
    elif url_input:
        if not is_valid_url(url_input):
            st.error("Invalid URL. Please enter a valid URL pointing to a .csv or .xlsx file.")
            st.stop()

        try:
            response = requests.get(url_input, timeout=10)
            if response.status_code == 200:
                if url_input.endswith(".csv"):
                    data = pd.read_csv(StringIO(response.text), parse_dates=["Date"], index_col="Date")
                elif url_input.endswith(".xlsx"):
                    data = pd.read_excel(StringIO(response.content), parse_dates=["Date"], index_col="Date")
            else:
                st.error(f"Error fetching data. Server returned status code: {response.status_code}")
                st.stop()
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.stop()

# Data column checker
if data is not None:
    close_column = detect_close_column(data)
    if close_column is None:
        st.error("No recognizable closing price column found.")
        st.stop()
    data = data[[close_column]].rename(columns={close_column: "Close"})
    data.dropna(inplace=True)
    data.sort_index(inplace=True)
else:
    st.warning("Please provide valid data.")
    st.stop()

# Create several tabs to split the product into sections
tabs = st.tabs(["Data", "Data Exploration", "Data Preparation", "Modeling", "Model Analysis", "Report Generation"])

# Tab 1 to let the user query the dataset for a closing price
with tabs[0]:
    st.header("Data")
    
    # Check index and set if needed
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Display the date range in the dataset
    min_date, max_date = data.index.min().date(), data.index.max().date()
    st.write(f"Stock closing prices are available from {min_date} to {max_date}")
    st.text("")
    
    # Find the closing price for the selected day if possible. Find for next available day otherwise.
    selected_date = st.date_input("Select Date:", min_value=min_date, max_value=max_date)
    selected_date = pd.Timestamp(selected_date)
    if selected_date < data.index.min() or selected_date > data.index.max():
        st.warning(f"The selected date {selected_date.date()} is out of the available date range. Please choose a valid date within the range.")
        selected_date = get_next_available_date(selected_date, data)
    if selected_date not in data.index:
        selected_date = get_next_available_date(selected_date, data)
    closing_price = get_closing_price(data, selected_date)
    st.write(f"Closing Price on {selected_date.date()}: ${closing_price:.2f}")

# Data Exploration: Display first and last few rows, generate summary statistics, and plots.
with tabs[1]:
    st.header("Data Exploration")

    # Table of first 10 rows
    with st.expander("First 10 Rows"):
        st.write(data.head(10))
    
    # Table of last 10 rows
    with st.expander("Last 10 Rows"):
        st.write(data.tail(10))

    # Table of Summary Statistics
    with st.expander("Summary Statistics"):
        st.write(data.describe())

    # List to store plots
    plots = {}

    # Line Plot
    fig1, ax1 = plt.subplots()
    ax1.plot(data.index, data['Close'])
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Closing Prices ($)")
    ax1.legend()
    ax1.set_title("Stock Price Over Time")
    plt.xticks(rotation=45)
    plots["Line Plot"] = fig1

    # Distribution Plot
    fig2, ax2 = plt.subplots()
    sns.histplot(data['Close'], kde = True, bins = 30, ax = ax2)
    ax2.set_xlabel("Closing Prices ($)")
    ax2.set_ylabel("Counts")
    ax2.set_title("Distribution")
    plots["Distribution Plot"] = fig2

    # Lag Plot
    fig3, ax3 = plt.subplots()
    pd.plotting.lag_plot(data['Close'], ax = ax3)
    ax3.set_xlabel("Closing Prices ($)")
    ax3.set_ylabel("Closing Prices ($)")
    ax3.set_title("Lag Plot")
    plots["Lag Plot"] = fig3
    
    # Plot selection
    plot_type = st.selectbox("Select Plot Type", list(plots.keys()))
    st.pyplot(plots[plot_type])

with tabs[2]:
    st.header("Data Preparation")

    # Check for missing values
    missing_values = data.isna().sum().sum()
    st.write(f"Total Missing Values: {missing_values}")
    if data.isnull().sum().sum() > 0:
        data.fillna(method='ffill', inplace=True)
        st.write("Missing Values have been forward filled.")

    # Check for number of outliers
    z_scores = (data['Close'] - data['Close'].mean()) / data['Close'].std()
    outliers = z_scores.abs() > 3
    st.write(f"Number of Outliers: {outliers.sum()}")

    # Scaling method selection
    scaling_method = st.selectbox("Select Scaling Method", ["Min-Max", "Standard", "Robust"])
    scaler = MinMaxScaler() if scaling_method == "Min-Max" else StandardScaler() if scaling_method == "Standard" else RobustScaler()
    data['Close_Scaled'] = scaler.fit_transform(data[['Close']])

    with st.expander("Before and after scaling"):
        scaling_table = pd.DataFrame({
            "Date": data.index,
            "Before": data['Close'].values,
            "Scaled": data['Close_Scaled'].values
        })
        st.write("Scaling")
        st.write(scaling_table.head())
    
with tabs[3]: 
    st.header("Modeling")

    # Set session state that model has not been trained yet
    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False

    # Make buttons to run and reset model
    col1, col2 = st.columns(2)
    with col1:
        run_model = st.button("Run Model")
    with col2:
        reset_model = st.button("Reset Model")

    if run_model:
        # Fit scaler only on scaled closing prices
        data_scaled = data.copy()
        data_scaled['Close_Scaled'] = scaler.fit_transform(data[['Close']])

        # Create sequences using scaled data with 80% train and 20% test
        seq_length = 30
        X, y = create_sequences(data_scaled['Close_Scaled'].values, seq_length)
        # Reshape y to be a 1D array
        y = y.reshape(-1)
        # Find the index corresponding to the split ratio and take slices since the data is time series
        train_size = int(len(X) * 0.8)
        X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

        # Define and train model
        model = Sequential([
            Input(shape=(X_train.shape[1], 1)),
            LSTM(32, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer="Adam", loss="mean_absolute_error")
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(X_train, 
                            y_train, 
                            epochs=50, 
                            batch_size=32, 
                            validation_data=(X_test, y_test), 
                            callbacks=[early_stopping])

        # Get predictions and revert the scaling
        y_pred = model.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Set the session state that the model has been trained
        st.session_state.model_trained = True
        st.session_state['y_test'] = y_test
        st.session_state['y_pred'] = y_pred
        st.success("Model trained successfully!")
    
    # Reset the model
    if reset_model:
        st.session_state.model_trained = False
        st.warning("Model has been reset. Adjust parameters and run again.")

    # Prompt to train the model if not
    if not st.session_state.model_trained:
        st.warning("Click 'Run Model' to train the model.")

# Plotting
with tabs[4]:
    st.header("Model Analysis")
    # Check if model has been run
    if not st.session_state.model_trained:
        st.warning("Click 'Run Model' to train the model.")
        st.stop()

    # Retrieve stored data
    y_test = st.session_state.get('y_test')
    y_pred = st.session_state.get('y_pred')

    if y_test is None or y_pred is None:
        st.error("Model outputs not found. Please run the model first.")
        st.stop()

    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)

    # Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # Mean Percentage Error (MPE)
    epsilon = 1e-8  # Small value to prevent division by zero
    mpe = np.mean((y_test - y_pred) / (y_test + epsilon)) * 100

    # Mean Absolute Percentage Error (MAPE)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    # R-Squared (R2)
    r2 = r2_score(y_test, y_pred)

   # Calculate performance metrics
    with st.expander("Performance Metrics"):
        st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
        st.write(f"Mean Squared Error (MSE): {mse:.4f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        st.write(f"Mean Percentage Error (MPE): {mpe:.4f}")
        st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
        st.write(f"R-squared: {r2:.4f}")

    # List to store model plots
    model_plots = {}

    # Predictions vs Actual
    fig4, ax4 = plt.subplots()
    ax4.plot(data.index[-len(y_test):], y_test, label = "Actual Prices", color = 'dodgerblue')
    ax4.plot(data.index[-len(y_test):], y_pred, label = "Predicted Prices", color = 'red')
    ax4.set_title("Predictions vs Actual")
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Stock Price ($)")
    ax4.legend()
    plt.xticks(rotation=45)
    model_plots["Predictions vs Actual"] = fig4

    # Predictions vs Actual (Last 100 days)
    fig5, ax5 = plt.subplots()
    ax5.plot(data.index[-100:], y_test[-100:], label=" Actual Prices", color="dodgerblue")
    ax5.plot(data.index[-100:], y_pred[-100:], label=" Predicted Prices", color="red")
    ax5.set_xlabel("Date")
    ax5.set_ylabel("Closing Prices ($)")
    ax5.legend()
    ax5.set_title("Predictions vs. Actual (Last 100 Days)")
    plt.xticks(rotation=45)
    model_plots["Predictions vs Actual (Last 100 days)"] = fig5

    # Compute directional movement
    actual_direction = np.sign(np.diff(y_test.squeeze()))  # Squeeze ensures proper shape
    predicted_direction = np.sign(np.diff(y_pred.squeeze()))

    # Remove NaN values if they exist
    actual_direction = actual_direction[~np.isnan(actual_direction)]
    predicted_direction = predicted_direction[~np.isnan(predicted_direction)]

    # Ensure arrays have the same length
    min_len = min(len(actual_direction), len(predicted_direction))
    actual_direction = actual_direction[:min_len]
    predicted_direction = predicted_direction[:min_len]

    # Counting correct and incorrect predictions, but excluding areas with no change
    if len(actual_direction) == 0 or len(predicted_direction) == 0:
        correct_predictions, incorrect_predictions = 0, 0  # Avoid errors
    else:
        mask = actual_direction != 0  # Ignore cases where there's no change
        correct_predictions = np.sum(actual_direction[mask] == predicted_direction[mask])
        incorrect_predictions = np.sum(actual_direction[mask] != predicted_direction[mask])

    # Ensure values are not NaN before plotting
    if np.isnan(correct_predictions) or np.isnan(incorrect_predictions):
        correct_predictions, incorrect_predictions = 0, 0

    # Plotting Directional Accuracy
    fig6, ax6 = plt.subplots()
    labels = ['Correct', 'Incorrect']
    counts = [correct_predictions, incorrect_predictions]

    # Avoid plotting an empty pie chart
    if sum(counts) == 0:
        st.warning("Directional accuracy data is empty. Cannot generate pie chart.")
    else:
        ax6.pie(counts, labels=labels, autopct='%1.1f%%', colors=['#2ECC71', '#E74C3C'])
        ax6.set_title('Directional Accuracy')
        model_plots["Directional Accuracy"] = fig6

    # Plot selection
    model_plot_type = st.selectbox("Select Plot Type", list(model_plots.keys()))
    st.pyplot(model_plots[model_plot_type])
    
with tabs[5]:
    st.header("Report Generation")
    # Checkbox to select Summary Statistics
    include_summary_stats = st.checkbox("Include Summary Statistics")

    # Checkbox to select Performance Metrics
    include_performance_metrics = st.checkbox("Include Performance Metrics")

    # Checkboxes for Plots
    selected_plot_keys = []
    st.write("Select Plots to Include:")

    # Performance Metrics
    performance_metrics = {
        "Mean Absolute Error (MAE)": mae,
        "Mean Squared Error (MSE)": mse,
        "Root Mean Squared Error (RMSE)": rmse,
        "Mean Percentage Error (MPE)": mpe,
        "Mean Absolute Percentage Error (MAPE)": mape,
        "R-squared": r2
    }

    combined_plot_keys = list(plots.keys()) + list(model_plots.keys())
    for plot_key in combined_plot_keys:
        if st.checkbox(plot_key):
            selected_plot_keys.append(plot_key)

    # Button to generate the report
    if st.button("Generate Report"):
        pdf = generate_pdf(include_summary_stats, include_performance_metrics, selected_plot_keys, plots, model_plots, performance_metrics)
        html = generate_html(include_summary_stats, include_performance_metrics, selected_plot_keys, plots, model_plots, performance_metrics)

        # Display HTML report
        st.components.v1.html(html, width=700, height=1000)

        # Provide download options
        st.success(f"Reports generated successfully!")

        # Button to download the generated PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            pdf.output(tmp_pdf.name)
            tmp_pdf.seek(0)
            st.download_button(
                label="Download PDF Report",
                data=tmp_pdf.read(),
                file_name="LSTM_Model_Report.pdf",
                mime="application/pdf"
                )
        # Button to download the generated HTML
        html_bytes = BytesIO(html.encode())
        html_bytes.seek(0)
        st.download_button(
            label="Download HTML Report",
            data=html_bytes,
            file_name="LSTM_Model_Report.html",
            mime="text/html"
        )
