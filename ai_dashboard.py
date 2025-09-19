import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ----------------------------
# Generate Synthetic Data
# ----------------------------
np.random.seed(42)
years = np.arange(2018, 2025)
months = pd.date_range("2018-01", "2024-12", freq="M")

# Simulate AI usage percentage and sales
data = pd.DataFrame({
    "Date": months,
    "AI_Usage": np.clip(np.linspace(10, 80, len(months)) + np.random.normal(0, 5, len(months)), 5, 95),
})

# Sales influenced by AI usage + seasonal effects + randomness
data["Sales"] = 50000 + data["AI_Usage"] * 1200 + 5000*np.sin(np.arange(len(months))/6) + np.random.normal(0, 5000, len(months))
data["Sales"] = data["Sales"].clip(lower=10000)

# Growth rate
monthly_growth = data["Sales"].pct_change().fillna(0) * 100
data["Sales_Growth(%)"] = monthly_growth.round(2)

# ----------------------------
# Train Predictive Model
# ----------------------------
X = data[["AI_Usage"]]
y = data["Sales"]
model = LinearRegression()
model.fit(X, y)

# Predict future sales for next 12 months
future_ai = np.linspace(data["AI_Usage"].iloc[-1], 95, 12)
future_dates = pd.date_range(data["Date"].iloc[-1] + pd.offsets.MonthBegin(1), periods=12, freq="M")
future_sales = model.predict(future_ai.reshape(-1, 1))

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "AI_Usage": future_ai,
    "Predicted_Sales": future_sales
})

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="AI Usage & Sales Dashboard", layout="wide")
st.title("📊 AI Usage and Sales Impact Dashboard")

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Current AI Usage %", f"{data['AI_Usage'].iloc[-1]:.1f}%")
col2.metric("Latest Monthly Sales", f"${data['Sales'].iloc[-1]:,.0f}")
col3.metric("Sales Growth (MoM)", f"{data['Sales_Growth(%)'].iloc[-1]:.2f}%")

# Line Chart: AI Usage vs Sales with Forecast
fig1 = px.line(data, x="Date", y="Sales", title="AI Usage vs Sales Over Time")
fig1.add_scatter(x=data["Date"], y=data["AI_Usage"]*1000, mode="lines", name="AI Usage (scaled)")
fig1.add_scatter(x=forecast_df["Date"], y=forecast_df["Predicted_Sales"], mode="lines+markers", name="Forecasted Sales")
st.plotly_chart(fig1, use_container_width=True)

# Scatterplot: AI Usage vs Sales
fig2 = px.scatter(data, x="AI_Usage", y="Sales",
                  trendline="ols",
                  title="Correlation: AI Usage vs Sales")
st.plotly_chart(fig2, use_container_width=True)

# Sales Growth Histogram
fig3 = px.histogram(data, x="Sales_Growth(%)", nbins=30,
                    title="Distribution of Monthly Sales Growth (%)")
st.plotly_chart(fig3, use_container_width=True)

# Forecast Table
st.subheader("🔮 Sales Forecast Based on AI Usage")
st.dataframe(forecast_df)

# Show Data Table with Filters
st.subheader("📋 Historical Data Table")
st.dataframe(data.tail(50))