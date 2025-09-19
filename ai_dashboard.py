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
regions = ["EMEA", "ASIA", "NAM"]
ai_agents = ["Pricing Assistant", "Deal Advisor", "Forecasting Bot", "Account Researcher", "Proposal Generator"]
sales_reps = ["Rep A", "Rep B", "Rep C", "Rep D", "Rep E"]

# Simulate AI usage percentage and sales
data = pd.DataFrame({
    "Date": months,
    "AI_Usage": np.clip(np.linspace(10, 80, len(months)) + np.random.normal(0, 5, len(months)), 5, 95),
    "Region": np.random.choice(regions, size=len(months)),
    "AI_Agent": np.random.choice(ai_agents, size=len(months)),
    "Sales_Rep": np.random.choice(sales_reps, size=len(months)),
    "AI_Supported": np.random.choice(["With AI", "Without AI"], size=len(months), p=[0.7, 0.3])
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
# Synthetic KPIs
# ----------------------------
win_more = 15.2  # % increase in deals won
win_rate = 48.5  # % overall win rate
avg_cycle = 62   # days
avg_deal_size = 87000  # USD
ai_win_rate = 57.8  # % influenced by AI

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="AI Usage & Sales Dashboard", layout="wide")
st.title("📊 AI Usage and Sales Impact Dashboard")

# KPIs
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Win More", f"{win_more:.1f}%")
col2.metric("Win Rate", f"{win_rate:.1f}%")
col3.metric("Avg Sales Cycle", f"{avg_cycle} days")
col4.metric("Avg Deal Size", f"${avg_deal_size:,.0f}")
col5.metric("AI Influenced Win Rate", f"{ai_win_rate:.1f}%")

# Existing KPIs
st.subheader("📌 Core AI & Sales Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Current AI Usage %", f"{data['AI_Usage'].iloc[-1]:.1f}%")
col2.metric("Latest Monthly Sales", f"${data['Sales'].iloc[-1]:,.0f}")
col3.metric("Sales Growth (MoM)", f"{data['Sales_Growth(%)'].iloc[-1]:.2f}%")

# Line Chart: AI Usage vs Sales with Forecast
fig1 = px.line(data, x="Date", y="Sales", title="AI Usage vs Sales Over Time", color_discrete_sequence=["#006771"])
fig1.add_scatter(x=data["Date"], y=data["AI_Usage"]*1000, mode="lines", name="AI Usage (scaled)", line=dict(color="#999999", dash="dot"))
fig1.add_scatter(x=forecast_df["Date"], y=forecast_df["Predicted_Sales"], mode="lines+markers", name="Forecasted Sales", line=dict(color="#006771", dash="dash"))
st.plotly_chart(fig1, use_container_width=True)

# Scatterplot: AI Usage vs Sales
fig2 = px.scatter(data, x="AI_Usage", y="Sales", color="Region",
                  title="Correlation: AI Usage vs Sales by Region",
                  color_discrete_sequence=["#006771", "#009688", "#004C4C"])
st.plotly_chart(fig2, use_container_width=True)

# Box & Whisker by Region
fig_box = px.box(data, x="Region", y="Sales", color="Region",
                 title="Sales Distribution by Region",
                 color_discrete_sequence=["#006771", "#009688", "#004C4C"])
st.plotly_chart(fig_box, use_container_width=True)

# Sales Growth Histogram
fig3 = px.histogram(data, x="Sales_Growth(%)", nbins=30,
                    title="Distribution of Monthly Sales Growth (%)",
                    color_discrete_sequence=["#006771"])
st.plotly_chart(fig3, use_container_width=True)

# ----------------------------
# AI Agent Analysis
# ----------------------------
st.header("🤖 AI Agent Performance Analysis")

agent_perf = data.groupby("AI_Agent").agg(
    Total_Sales=("Sales", "sum"),
    Avg_Deal_Size=("Sales", "mean"),
    Deals_Closed=("Sales", "count"),
    Avg_AI_Usage=("AI_Usage", "mean")
).reset_index()

# Bar chart: Top AI Agents by Total Sales
fig_agents = px.bar(
    agent_perf.sort_values("Total_Sales", ascending=False),
    x="AI_Agent", y="Total_Sales",
    text_auto=True,
    color="AI_Agent",
    title="Top AI Agents by Total Sales Impact",
    color_discrete_sequence=["#006771", "#009688", "#004C4C", "#33A6A6", "#80CBC4"]
)
st.plotly_chart(fig_agents, use_container_width=True)

# Box plot: Sales distribution per AI Agent
fig_agent_box = px.box(
    data, x="AI_Agent", y="Sales", color="AI_Agent",
    title="Sales Distribution per AI Agent",
    color_discrete_sequence=["#006771", "#009688", "#004C4C", "#33A6A6", "#80CBC4"]
)
st.plotly_chart(fig_agent_box, use_container_width=True)

# Agent performance table
st.subheader("📋 AI Agent Performance Metrics")
st.dataframe(agent_perf)

# ----------------------------
# Sales Rep With vs Without AI
# ----------------------------
st.header("👥 Sales Rep Performance: With vs Without AI Support")

rep_perf = data.groupby(["Sales_Rep", "AI_Supported"]).agg(
    Total_Sales=("Sales", "sum"),
    Avg_Deal_Size=("Sales", "mean"),
    Deals_Closed=("Sales", "count")
).reset_index()

fig_rep = px.bar(
    rep_perf, x="Sales_Rep", y="Total_Sales", color="AI_Supported",
    barmode="group", text_auto=True,
    title="Sales Rep Performance Comparison (With vs Without AI)",
    color_discrete_sequence=["#006771", "#999999"]
)
st.plotly_chart(fig_rep, use_container_width=True)

st.subheader("📋 Sales Rep Metrics by AI Support")
st.dataframe(rep_perf)

# Forecast Table
st.subheader("🔮 Sales Forecast Based on AI Usage")
st.dataframe(forecast_df)

# Show Data Table with Filters
st.subheader("📋 Historical Data Table")
st.dataframe(data.tail(50))
