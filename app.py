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
# Order-Level Risk Prediction
# -----------------------------
st.header("🔮 Order-Level Risk Prediction")

st.write(
    "Enter key order details to estimate the probability of late delivery."
)

with st.form("order_prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        input_shipping_mode = st.selectbox(
            "Shipping Mode",
            sorted(prediction_df["Shipping Mode"].dropna().unique())
        )

        input_market = st.selectbox(
            "Market",
            sorted(prediction_df["Market"].dropna().unique())
        )

    with col2:
        input_segment = st.selectbox(
            "Customer Segment",
            sorted(prediction_df["Customer Segment"].dropna().unique())
        )

        input_scheduled_days = st.number_input(
            "Scheduled Shipping Days",
            min_value=0,
            max_value=10,
            value=2,
            step=1
        )

    with col3:
        input_quantity = st.number_input(
            "Order Item Quantity",
            min_value=1,
            max_value=20,
            value=1,
            step=1
        )

        input_benefit = st.number_input(
            "Benefit per Order",
            value=0.0,
            step=1.0
        )

    predict_button = st.form_submit_button(
        "🚀 Predict Delivery Risk"
    )


if predict_button:

    # Start with a real row so every model-required
    # feature is present
    input_row = prediction_df.drop(
        columns=[
            "Predicted_Probability",
            "Predicted_Risk"
        ],
        errors="ignore"
    ).iloc[[0]].copy()

    # Replace user-selected values
    input_row["Shipping Mode"] = input_shipping_mode
    input_row["Market"] = input_market
    input_row["Customer Segment"] = input_segment
    input_row["Days for shipment (scheduled)"] = input_scheduled_days
    input_row["Order Item Quantity"] = input_quantity
    input_row["Benefit per order"] = input_benefit

    # Recalculate engineered features
    input_row["Shipping_Pressure_Index"] = (
        input_row["Order Item Quantity"] /
        (input_row["Days for shipment (scheduled)"] + 1)
    )

    input_row["Express_Mode_Flag"] = (
        input_row["Shipping Mode"]
        .isin(["First Class", "Same Day"])
        .astype(int)
    )

    # Predict probability
    individual_probability = model.predict_proba(
        input_row
    )[:, 1][0]

    # Risk category
    if individual_probability <= 0.33:
        individual_risk = "🟢 Low Risk"
        recommendation = (
            "Normal monitoring is sufficient."
        )

    elif individual_probability <= 0.66:
        individual_risk = "🟡 Medium Risk"
        recommendation = (
            "Monitor the shipment closely and "
            "consider prioritisation."
        )

    else:
        individual_risk = "🔴 High Risk"
        recommendation = (
            "Immediate intervention should be considered "
            "to reduce the possibility of delay."
        )

    # Display result
    st.markdown("---")
    st.subheader("📌 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Predicted Late Delivery Probability",
            f"{individual_probability * 100:.2f}%"
        )

    with result_col2:
        st.metric(
            "Risk Category",
            individual_risk
        )

    st.info(
        f"💡 **Recommended Action:** {recommendation}"
    )

    # Key risk indicators
    st.subheader("🔍 Key Risk Indicators")

    if input_scheduled_days <= 1:
        st.warning(
            "Short scheduled shipping time may increase delivery pressure."
        )

    if input_shipping_mode == "First Class":
        st.warning(
            "First Class shipping is associated with elevated "
            "late-delivery risk in the dataset."
        )

    if input_quantity >= 5:
        st.warning(
            "Higher order quantity increases the shipping "
            "pressure index."
        )

    if input_shipping_mode == "Same Day":
        st.success(
            "Same Day shipping selected."
        )

    if input_scheduled_days >= 4:
        st.success(
            "A longer scheduled shipping window may provide "
            "more operational flexibility."
        )

    st.markdown("---")
    # -----------------------------
# Operational Risk Insights
# -----------------------------
st.header("💡 Operational Risk Insights")

if len(filtered_df) > 0:

    highest_mode = (
        filtered_df.groupby("Shipping Mode")["Predicted_Probability"]
        .mean()
        .idxmax()
    )

    highest_mode_risk = (
        filtered_df.groupby("Shipping Mode")["Predicted_Probability"]
        .mean()
        .max() * 100
    )

    highest_region = (
        filtered_df.groupby("Order Region")["Predicted_Probability"]
        .mean()
        .idxmax()
    )

    highest_region_risk = (
        filtered_df.groupby("Order Region")["Predicted_Probability"]
        .mean()
        .max() * 100
    )

    highest_segment = (
        filtered_df.groupby("Customer Segment")["Predicted_Probability"]
        .mean()
        .idxmax()
    )

    immediate_orders = (
        filtered_df["Predicted_Probability"] >= 0.85
    ).sum()

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        st.info(
            f"🚚 **Highest-risk shipping mode:** {highest_mode} "
            f"with an average predicted risk of "
            f"**{highest_mode_risk:.1f}%**."
        )

        st.info(
            f"📍 **Highest-risk region:** {highest_region} "
            f"with an average predicted risk of "
            f"**{highest_region_risk:.1f}%**."
        )

    with insight_col2:
        st.info(
            f"👥 **Highest-risk customer segment:** "
            f"{highest_segment}."
        )

        st.warning(
            f"🚨 **Immediate intervention queue:** "
            f"{immediate_orders:,} orders have predicted risk "
            f"of **85% or above**."
        )

    st.markdown(
        "These insights help operations teams prioritize "
        "high-risk shipping modes, regions, customer segments, "
        "and individual orders for proactive intervention."
    )

else:
    st.warning("No orders match the selected filters.")
# -----------------------------
# High-Risk Order Contribution
# -----------------------------
st.header("📌 High-Risk Order Contribution")

if len(filtered_df) > 0:

    contribution_df = filtered_df[
        filtered_df["Predicted_Probability"] >= risk_threshold
    ].copy()

    if len(contribution_df) > 0:

        contribution = (
            contribution_df
            .groupby("Shipping Mode")
            .size()
            .reset_index(name="High_Risk_Orders")
        )

        contribution["Contribution_%"] = (
            contribution["High_Risk_Orders"]
            / contribution["High_Risk_Orders"].sum()
            * 100
        )

        contribution = contribution.sort_values(
            "Contribution_%",
            ascending=False
        )

        fig = px.bar(
            contribution,
            x="Shipping Mode",
            y="Contribution_%",
            text="Contribution_%",
            title="Contribution of Shipping Modes to High-Risk Orders"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig.update_layout(
            yaxis_title="Contribution to High-Risk Orders (%)",
            xaxis_title="Shipping Mode"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        top_contributor = contribution.iloc[0]

        st.info(
            f"🚚 **{top_contributor['Shipping Mode']}** contributes "
            f"the largest share of high-risk orders, accounting for "
            f"**{top_contributor['Contribution_%']:.1f}%** of the "
            f"high-risk orders under the current filters."
        )

    else:
        st.success(
            "No high-risk orders are present under the selected filters."
        )

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
# Model Evaluation
# -----------------------------
st.header("📊 Model Evaluation")

st.write(
    "The trained XGBoost model is evaluated using standard classification "
    "metrics on the test dataset."
)

try:
    # Load saved evaluation results
    metrics_df = pd.read_csv("model_metrics.csv")
    confusion_df = pd.read_csv("confusion_matrix.csv", index_col=0)

    # -----------------------------
    # Evaluation Metrics
    # -----------------------------
    st.subheader("📈 Classification Performance")

    metric_cols = st.columns(5)

    for col, (_, row) in zip(metric_cols, metrics_df.iterrows()):
        with col:
            col.metric(
                row["Metric"],
                f'{row["Score"]:.2%}'
            )

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    st.subheader("🔍 Confusion Matrix")

    fig_cm = px.imshow(
        confusion_df,
        text_auto=True,
        aspect="auto",
        labels={
            "x": "Predicted Class",
            "y": "Actual Class",
            "color": "Orders"
        },
        title="XGBoost Confusion Matrix"
    )

    st.plotly_chart(fig_cm, use_container_width=True)

    # -----------------------------
    # Business Interpretation
    # -----------------------------
    st.subheader("📌 Business Interpretation")

    st.markdown("""
    **Accuracy (70.32%)**  
    The model correctly classifies approximately 70% of the orders.

    **Precision (79.29%)**  
    When the model flags an order as late-risk, approximately 79% of the
    flagged orders are actually late. This helps reduce unnecessary
    operational interventions.

    **Recall (62.08%)**  
    The model identifies approximately 62% of the actual late deliveries.
    This means some late orders may remain undetected.

    **F1 Score (69.64%)**  
    The F1 score provides a balance between precision and recall and
    indicates a moderate overall classification performance.

    **ROC-AUC (77.27%)**  
    The ROC-AUC indicates that the model has good ability to distinguish
    between late-risk and on-time orders.
    """)

    # -----------------------------
    # Operational Interpretation
    # -----------------------------
    st.subheader("🚚 Operational Implication")

    st.info(
        "The model can be used as an early-warning system to prioritize "
        "orders that require operational attention. Higher-risk orders "
        "can be reviewed first for proactive intervention such as "
        "rerouting, shipment prioritization, or customer communication."
    )

except Exception:
    st.warning(
        "Model evaluation files could not be loaded. "
        "Please ensure model_metrics.csv and confusion_matrix.csv "
        "are present in the project repository."
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
