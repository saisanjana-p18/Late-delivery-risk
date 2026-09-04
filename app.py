import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Late Delivery Risk Prediction",
    page_icon="🚚",
    layout="wide"
)

# -----------------------------
# Load data and model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("xgb_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("dashboard_data.csv", encoding="latin1")

model = load_model()
df = load_data()

# -----------------------------
# Prepare prediction data
# -----------------------------
prediction_df = df.copy()

prediction_df = prediction_df.drop(
    columns=[
        "Late_delivery_risk",
        "Days for shipping (real)",
        "Delivery Status",
        "Order Status"
    ],
    errors="ignore"
)

prediction_df = prediction_df.drop(
    columns=[
        "Customer Fname",
        "Customer Lname",
        "Customer Street",
        "Customer Id",
        "Order Customer Id",
        "Category Id",
        "Department Id",
        "Customer Zipcode"
    ],
    errors="ignore"
)

prediction_df = prediction_df.drop(
    columns=[
        "Customer City",
        "Customer State",
        "Order City",
        "Order State",
        "Product Name"
    ],
    errors="ignore"
)

prediction_df["Shipping_Pressure_Index"] = (
    prediction_df["Order Item Quantity"] /
    (prediction_df["Days for shipment (scheduled)"] + 1)
)

prediction_df["Express_Mode_Flag"] = (
    prediction_df["Shipping Mode"]
    .isin(["First Class", "Same Day"])
    .astype(int)
)

# -----------------------------
# Predictions
# -----------------------------
prediction_df["Predicted_Probability"] = model.predict_proba(
    prediction_df
)[:, 1]

prediction_df["Predicted_Risk"] = pd.cut(
    prediction_df["Predicted_Probability"],
    bins=[-0.01, 0.33, 0.66, 1.0],
    labels=["Low", "Medium", "High"]
)

# -----------------------------
# Header
# -----------------------------
st.title("🚚 Late Delivery Risk Prediction Dashboard")
st.markdown(
    "### Machine Learning–based Late Delivery Risk Prediction "
    "in Global Supply Chain Operations"
)

st.markdown("---")

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("🎚️ Dashboard Filters")

shipping_modes = ["All"] + sorted(
    df["Shipping Mode"].dropna().unique().tolist()
)

selected_mode = st.sidebar.selectbox(
    "Shipping Mode",
    shipping_modes
)

markets = ["All"] + sorted(
    df["Market"].dropna().unique().tolist()
)

selected_market = st.sidebar.selectbox(
    "Market",
    markets
)

segments = ["All"] + sorted(
    df["Customer Segment"].dropna().unique().tolist()
)

selected_segment = st.sidebar.selectbox(
    "Customer Segment",
    segments
)

risk_threshold = st.sidebar.slider(
    "High-Risk Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.66,
    step=0.01
)

# -----------------------------
# Apply filters
# -----------------------------
filtered_df = prediction_df.copy()

if selected_mode != "All":
    filtered_df = filtered_df[
        filtered_df["Shipping Mode"] == selected_mode
    ]

if selected_market != "All":
    filtered_df = filtered_df[
        filtered_df["Market"] == selected_market
    ]

if selected_segment != "All":
    filtered_df = filtered_df[
        filtered_df["Customer Segment"] == selected_segment
    ]

# -----------------------------
# KPI calculations
# -----------------------------
total_orders = len(filtered_df)

high_risk_orders = (
    filtered_df["Predicted_Probability"] >= risk_threshold
).sum()

average_risk = (
    filtered_df["Predicted_Probability"].mean()
    if total_orders > 0 else 0
)

historical_late_rate = (
    df.loc[filtered_df.index, "Late_delivery_risk"].mean() * 100
    if total_orders > 0 else 0
)

# -----------------------------
# KPI cards
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "📦 Total Orders",
    f"{total_orders:,}"
)

col2.metric(
    "🔴 High-Risk Orders",
    f"{high_risk_orders:,}"
)

col3.metric(
    "🎯 Average Predicted Risk",
    f"{average_risk * 100:.2f}%"
)

col4.metric(
    "📊 Historical Late Rate",
    f"{historical_late_rate:.2f}%"
)

st.markdown("---")

# -----------------------------
# Risk Overview
# -----------------------------
st.header("📊 Delay Risk Overview")

risk_counts = (
    filtered_df["Predicted_Risk"]
    .value_counts()
    .reindex(["Low", "Medium", "High"])
    .fillna(0)
    .reset_index()
)

risk_counts.columns = ["Risk Level", "Orders"]

col1, col2 = st.columns(2)

with col1:
    fig = px.pie(
        risk_counts,
        names="Risk Level",
        values="Orders",
        title="Predicted Risk Distribution",
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.bar(
        risk_counts,
        x="Risk Level",
        y="Orders",
        title="Orders by Risk Level",
        text="Orders"
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Shipping Mode Analysis
# -----------------------------
st.header("🚚 Shipping Mode Risk Analysis")

mode_analysis = (
    filtered_df.groupby("Shipping Mode")
    .agg(
        Orders=("Predicted_Probability", "count"),
        Average_Risk=("Predicted_Probability", "mean")
    )
    .reset_index()
)

mode_analysis["Average_Risk"] *= 100

fig = px.bar(
    mode_analysis,
    x="Shipping Mode",
    y="Average_Risk",
    text="Average_Risk",
    title="Average Predicted Risk by Shipping Mode"
)

fig.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Market Analysis
# -----------------------------
st.header("🌎 Market Risk Analysis")

market_analysis = (
    filtered_df.groupby("Market")
    .agg(
        Orders=("Predicted_Probability", "count"),
        Average_Risk=("Predicted_Probability", "mean")
    )
    .reset_index()
)

market_analysis["Average_Risk"] *= 100

fig = px.bar(
    market_analysis,
    x="Market",
    y="Average_Risk",
    text="Average_Risk",
    title="Average Predicted Risk by Market"
)

fig.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Regional Analysis
# -----------------------------
st.header("📍 Regional Risk Analysis")

region_analysis = (
    filtered_df.groupby("Order Region")
    .agg(
        Orders=("Predicted_Probability", "count"),
        Average_Risk=("Predicted_Probability", "mean")
    )
    .reset_index()
)

region_analysis["Average_Risk"] *= 100

fig = px.bar(
    region_analysis.sort_values(
        "Average_Risk",
        ascending=False
    ),
    x="Average_Risk",
    y="Order Region",
    orientation="h",
    title="Risk by Order Region"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Customer Segment Analysis
# -----------------------------
st.header("👥 Customer Segment Risk")

segment_analysis = (
    filtered_df.groupby("Customer Segment")
    .agg(
        Orders=("Predicted_Probability", "count"),
        Average_Risk=("Predicted_Probability", "mean")
    )
    .reset_index()
)

segment_analysis["Average_Risk"] *= 100

fig = px.bar(
    segment_analysis,
    x="Customer Segment",
    y="Average_Risk",
    text="Average_Risk",
    title="Average Predicted Risk by Customer Segment"
)

fig.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# High-risk operations panel
# -----------------------------
st.header("⚠️ Operations Action Panel")

high_risk_df = filtered_df[
    filtered_df["Predicted_Probability"] >= risk_threshold
].copy()

if len(high_risk_df) > 0:

    high_risk_df["Recommended Action"] = np.where(
        high_risk_df["Predicted_Probability"] >= 0.85,
        "🚨 Immediate intervention / expedite shipment",
        np.where(
            high_risk_df["Predicted_Probability"] >= 0.66,
            "⚠️ Closely monitor and prioritize",
            "👀 Monitor shipment"
        )
    )

    display_columns = [
        "Shipping Mode",
        "Market",
        "Order Region",
        "Customer Segment",
        "Predicted_Probability",
        "Predicted_Risk",
        "Recommended Action"
    ]

    display_columns = [
        c for c in display_columns
        if c in high_risk_df.columns
    ]

    action_table = high_risk_df[
        display_columns
    ].sort_values(
        "Predicted_Probability",
        ascending=False
    ).head(100)

    action_table["Predicted_Probability"] = (
        action_table["Predicted_Probability"] * 100
    ).round(2)

    st.dataframe(
        action_table,
        use_container_width=True
    )

else:
    st.success(
        "🎉 No orders currently exceed the selected risk threshold."
    )

# -----------------------------
# Model Explainability
# -----------------------------
st.header("🧠 Model Explainability")

try:

    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()
    importance = classifier.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    importance_df = (
        importance_df
        .sort_values("Importance", ascending=False)
        .head(15)
    )

    importance_df["Feature"] = (
        importance_df["Feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )

    fig = px.bar(
        importance_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Features Influencing Late Delivery Risk"
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:

    st.warning(
        "Feature importance could not be displayed for this model."
    )

# -----------------------------
# Model Information
# -----------------------------
st.markdown("---")

st.header("🤖 Model Information")

st.write("""
**Model:** XGBoost Classifier

**Purpose:** Predict the probability that an order will experience a late delivery.

**Risk Categories:**
- 🟢 Low Risk: 0–33%
- 🟡 Medium Risk: 33–66%
- 🔴 High Risk: above 66%

The dashboard supports proactive identification and prioritisation
of potentially delayed shipments.
""")

st.success(
    "Dashboard successfully loaded using the trained XGBoost model."
)
