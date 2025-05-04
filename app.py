import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Streamlit page config
st.set_page_config(page_title="Financial ML App", layout="centered")
st.title("📈 Financial Machine Learning App")
st.write("Upload a dataset or fetch from Yahoo Finance to predict closing prices using Linear Regression.")

# Sidebar for data source selection
st.sidebar.header("📊 Select Data Source")
data_source = st.sidebar.radio("Choose:", ["Upload CSV", "Yahoo Finance"])

df = None  # Placeholder

# --- Option 1: Upload CSV ---
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded!")

# --- Option 2: Yahoo Finance ---
elif data_source == "Yahoo Finance":
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL")
    start = st.sidebar.date_input("Start Date")
    end = st.sidebar.date_input("End Date")
    if st.sidebar.button("Fetch Data"):
        df = yf.download(ticker, start=start, end=end)

        # ✅ Flatten MultiIndex columns (important!)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            st.error("⚠️ No data found. Try another symbol or date.")
        else:
            st.success("✅ Yahoo Finance data loaded!")

# --- If Data is Loaded ---
if df is not None and not df.empty:
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # Drop missing values
    df.dropna(inplace=True)

    # Safe, independent features only
    safe_features = ["Open", "High", "Low", "Volume"]
    target_column = "Close"

    # Check required columns
    missing = [col for col in safe_features + [target_column] if col not in df.columns]
    if missing:
        st.error(f"❌ Missing columns: {', '.join(missing)}")
        st.stop()

    # Assign features and label
    X = df[safe_features]
    y = df[target_column]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation Metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # Display metrics
    st.subheader("📈 Model Performance")
    st.metric("R² Score", f"{r2:.4f}")
    st.metric("Mean Squared Error", f"{mse:.2f}")

    # Visualization
    st.subheader("📉 Actual vs Predicted")
    fig, ax = plt.subplots()
    ax.plot(y_test.values, label="Actual", linewidth=2)
    ax.plot(y_pred, label="Predicted", linewidth=2)
    ax.set_title("Actual vs Predicted Close Prices")
    ax.legend()
    st.pyplot(fig)
else:
    st.info("📂 Upload a CSV or fetch data to continue.")
