import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Nassau Candy Demand Forecasting",
    page_icon="🍫",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, "data", "nassau_candy_cleaned.csv"))

@st.cache_data
def load_results():
    path = os.path.join(BASE_DIR, "outputs", "results")
    return {
        "kpis": pd.read_csv(os.path.join(path, "project_kpis.csv")),
        "evaluation": pd.read_csv(os.path.join(path, "model_evaluation.csv")),
        "cv": pd.read_csv(os.path.join(path, "cross_validation.csv")),
        "recommendations": pd.read_csv(os.path.join(path, "all_recommendations.csv")),
        "slow": pd.read_csv(os.path.join(path, "top_slow_products.csv")),
        "profitable": pd.read_csv(os.path.join(path, "top_profitable_products.csv")),
    }

@st.cache_resource
def load_context():
    return joblib.load(os.path.join(BASE_DIR, "models", "scenario_context.joblib"))

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, "models", "best_model.joblib"))

df = load_data()
results = load_results()
context = load_context()

st.title("🍫 Nassau Candy Demand Forecasting & Supply Optimization")
st.caption("Machine Learning dashboard for lead-time prediction, model evaluation and factory reassignment analysis.")

# Product/factory mappings created during the project
product_division = context.get("product_division", {})
product_factory = context.get("product_factory", {})
factories = list(context.get("factories", {}).keys())
features = context.get("features", [])

# Sidebar filters
st.sidebar.header("Dashboard Filters")
regions = sorted(df["Region"].dropna().unique())
divisions = sorted(df["Division"].dropna().unique())
ship_modes = sorted(df["Ship Mode"].dropna().unique())

selected_regions = st.sidebar.multiselect("Region", regions, default=regions)
selected_divisions = st.sidebar.multiselect("Division", divisions, default=divisions)
filtered = df[
    df["Region"].isin(selected_regions) &
    df["Division"].isin(selected_divisions)
].copy()

st.sidebar.markdown("---")
st.sidebar.info("Use the tabs to explore the dataset, model performance and supply-chain recommendations.")

# Top KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Orders / Rows", f"{len(filtered):,}")
c2.metric("Total Sales", f"${filtered['Sales'].sum():,.2f}")
c3.metric("Gross Profit", f"${filtered['Gross Profit'].sum():,.2f}")
c4.metric("Avg Lead Time", f"{filtered['Lead Time Days'].mean():,.1f} days")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", "📦 Dataset Overview", "🤖 Model Performance",
    "🔮 Lead-Time Prediction", "🏭 Recommendations"
])

with tab1:
    st.subheader("Sales & Operations Overview")

    left, right = st.columns(2)

    with left:
        sales_div = (
            filtered.groupby("Division", as_index=False)["Sales"]
            .sum()
            .sort_values("Sales", ascending=False)
        )
        fig = px.bar(sales_div, x="Division", y="Sales",
                     title="Sales by Division", text_auto=".2s")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        region_sales = (
            filtered.groupby("Region", as_index=False)["Sales"]
            .sum()
            .sort_values("Sales", ascending=False)
        )
        fig = px.bar(region_sales, x="Region", y="Sales",
                     title="Sales by Region", text_auto=".2s")
        st.plotly_chart(fig, use_container_width=True)

    monthly = filtered.copy()
    monthly["Order Date"] = pd.to_datetime(monthly["Order Date"], errors="coerce")
    monthly = (
        monthly.dropna(subset=["Order Date"])
        .set_index("Order Date")
        .resample("ME")["Sales"]
        .sum()
        .reset_index()
    )
    fig = px.line(monthly, x="Order Date", y="Sales",
                  markers=True, title="Monthly Sales Trend")
    st.plotly_chart(fig, use_container_width=True)

    a, b, c = st.columns(3)
    a.metric("Avg Profit Margin", f"{filtered['Profit Margin'].mean()*100:.2f}%")
    b.metric("Avg Units / Order", f"{filtered['Units'].mean():.2f}")
    c.metric("Avg Sales / Unit", f"${filtered['Sales Per Unit'].mean():.2f}")

with tab2:
    st.subheader("Cleaned Dataset")
    st.write(f"Showing {len(filtered):,} filtered rows from {len(df):,} total rows.")
    st.dataframe(filtered.head(1000), use_container_width=True, height=450)

    st.subheader("Top Slow Products")
    st.dataframe(results["slow"], use_container_width=True)

    st.subheader("Top Profitable Products")
    st.dataframe(results["profitable"], use_container_width=True)

with tab3:
    st.subheader("Model Evaluation")

    evaluation = results["evaluation"]
    st.dataframe(evaluation, use_container_width=True)

    fig = px.bar(
        evaluation, x="Model", y="RMSE",
        title="Model Comparison — RMSE (Lower is Better)",
        text_auto=".2f"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        evaluation, x="Model", y="R2",
        title="Model Comparison — R² (Higher is Better)",
        text_auto=".3f"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cross-Validation Results")
    st.dataframe(results["cv"], use_container_width=True)

    best_name = context.get("best_model_name", "Best Model")
    st.success(f"Selected model from the completed project: **{best_name}**")

with tab4:
    st.subheader("Predict Lead Time for a Scenario")
    st.write("Enter an order scenario and the saved project model will estimate lead time.")

    products = sorted(product_division.keys())
    if not products:
        products = sorted(df["Product Name"].dropna().unique())

    p1, p2 = st.columns(2)
    with p1:
        product = st.selectbox("Product", products)
        units = st.number_input("Units", min_value=1, value=2, step=1)
        sales = st.number_input("Sales", min_value=0.0, value=10.0, step=0.5)
        cost = st.number_input("Cost", min_value=0.0, value=5.0, step=0.5)

    with p2:
        region = st.selectbox("Region", regions)
        ship_mode = st.selectbox("Ship Mode", ship_modes)
        default_factory = product_factory.get(product, factories[0] if factories else "")
        factory_options = factories if factories else sorted(df.get("Factory", pd.Series(dtype=str)).dropna().unique())
        factory = st.selectbox(
            "Factory",
            factory_options,
            index=factory_options.index(default_factory) if default_factory in factory_options else 0
        )

    info = product_division.get(product, {})
    product_id = info.get("Product ID", "")
    division = info.get("Division", "")

    if st.button("🔮 Predict Lead Time", type="primary"):
        input_df = pd.DataFrame([{
            "Product ID": product_id,
            "Division": division,
            "Factory": factory,
            "Region": region,
            "Ship Mode": ship_mode,
            "Units": units,
            "Sales": sales,
            "Cost": cost
        }])

        try:
            model = load_model()
            prediction = float(model.predict(input_df)[0])
            st.metric("Predicted Lead Time", f"{prediction:,.1f} days")
            st.caption("Prediction generated by the saved Linear Regression pipeline from the completed project.")
        except Exception as e:
            st.error("The saved model could not be loaded in this environment.")
            st.code(str(e))
            st.info("Make sure requirements.txt uses scikit-learn==1.6.1, matching the version used to create the model.")

with tab5:
    st.subheader("Factory Reassignment Recommendations")

    rec = results["recommendations"].copy()

    r1, r2, r3 = st.columns(3)
    with r1:
        rec_region = st.selectbox("Recommendation Region", ["All"] + sorted(rec["Region"].dropna().unique()))
    with r2:
        rec_ship = st.selectbox("Recommendation Ship Mode", ["All"] + sorted(rec["Ship Mode"].dropna().unique()))
    with r3:
        risk = st.selectbox("Risk Flag", ["All"] + sorted(rec["Risk Flag"].dropna().unique()))

    view = rec.copy()
    if rec_region != "All":
        view = view[view["Region"] == rec_region]
    if rec_ship != "All":
        view = view[view["Ship Mode"] == rec_ship]
    if risk != "All":
        view = view[view["Risk Flag"] == risk]

    view = view.sort_values("Recommendation Score", ascending=False)

    st.metric("Recommendations Found", f"{len(view):,}")

    display_cols = [
        "Product", "Current Factory", "Candidate Factory", "Region",
        "Ship Mode", "Predicted Lead Time Days", "Lead Time Reduction Days",
        "Lead Time Reduction %", "Risk SD Days", "Estimated Profit",
        "Profit Impact %", "Recommendation Score", "Recommendation", "Risk Flag"
    ]
    st.dataframe(view[display_cols].head(500), use_container_width=True, height=500)

    if len(view):
        fig = px.scatter(
            view.head(200),
            x="Lead Time Reduction %",
            y="Recommendation Score",
            size="Estimated Profit",
            hover_name="Product",
            color="Risk Flag",
            title="Recommendation Score vs Lead-Time Reduction"
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Nassau Candy Distributor | Machine Learning & Supply-Chain Optimization Project")
