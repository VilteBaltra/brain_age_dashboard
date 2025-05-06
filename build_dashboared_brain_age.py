import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st

# ---- Page config ----
st.set_page_config(page_title="Brain Age Dashboard", layout="wide")

st.title("🧠 Brain Age Dashboard")
st.markdown("Explore predicted brain ages from multiple models, Brain-PADs, and age-related trends in your data.")

REQUIRED_COLUMNS = ["participant_id", "chronological_age", "sex"] + [f"predicted_brain_age_model{i}" for i in range(1, 6)]

@st.cache_data
def load_default_data():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "participant_id": [f"subj_{i+1}" for i in range(n)],
        "chronological_age": np.random.randint(20, 80, n),
        "sex": np.random.choice(["Male", "Female"], n)
    })
    
    # Model 1: A good model, closely matching chronological age
    df["predicted_brain_age_model1"] = df["chronological_age"] + np.random.normal(0, 3, n)  # Small noise
    
    # Model 2: A decelerated age prediction for older individuals
    df["predicted_brain_age_model2"] = df["chronological_age"] - (0.1 * df["chronological_age"]) + np.random.normal(0, 5, n)
    
    # Model 3: An accelerated age prediction for older individuals
    df["predicted_brain_age_model3"] = df["chronological_age"] + (0.1 * df["chronological_age"]) + np.random.normal(0, 5, n)
    
    # Model 4: A constant offshoot by +10 years
    df["predicted_brain_age_model4"] = df["chronological_age"] + 10 + np.random.normal(0, 3, n)
    
    # Model 5: A random model (poor prediction)
    df["predicted_brain_age_model5"] = np.random.randint(20, 80, n) + np.random.normal(0, 15, n)  # Large noise

    return df

# ---- Sidebar File Uploader ----
st.sidebar.header("📁 Upload Your Own Data (Optional)")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}. Loading default data instead.")
            df = load_default_data()
        else:
            st.success("Successfully loaded your data!")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        df = load_default_data()
else:
    df = load_default_data()

# ---- Preprocess ----
model_cols = [col for col in df.columns if col.startswith("predicted_brain_age_model")]
for model in model_cols:
    df[f"brain_PAD_{model}"] = df[model] - df["chronological_age"]

# ---- Sidebar filters ----
with st.sidebar:
    st.header("🔍 Filter Options")
    selected_sex = st.multiselect("Sex", df["sex"].unique(), default=list(df["sex"].unique()))
    age_min, age_max = st.slider("Chronological Age Range", 
                                 int(df["chronological_age"].min()), 
                                 int(df["chronological_age"].max()), 
                                 (20, 80))
    selected_models = st.multiselect("Select Model(s)", model_cols, default=model_cols)

df_filtered = df[
    (df["sex"].isin(selected_sex)) &
    (df["chronological_age"].between(age_min, age_max))
]


# ---- Summary Statistics ----
st.subheader("📊 Summary Statistics by Model")
summary_data = []
for model in selected_models:
    predicted_col = model
    brain_pad_col = f"brain_PAD_{model}"

    # Mean Predicted Age
    mean_predicted_age = df_filtered[predicted_col].mean()

    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(df_filtered[predicted_col] - df_filtered["chronological_age"]))

    # Mean Brain-PAD
    mean_brain_pad = df_filtered[brain_pad_col].mean()

    # Correlation between Predicted Age and Chronological Age
    corr = df_filtered[[predicted_col, "chronological_age"]].corr().iloc[0, 1]

    # Add to the summary data list
    summary_data.append({
        "Model": model,
        "Mean Predicted Age": mean_predicted_age,
        "MAE": mae,
        "Mean Brain-PAD": mean_brain_pad,
        "Correlation (Predicted vs Chronological Age)": corr
    })

summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df.style.format({
    "Mean Predicted Age": "{:.2f}",
    "MAE": "{:.2f}",
    "Mean Brain-PAD": "{:.2f}",
    "Correlation (Predicted vs Chronological Age)": "{:.2f}"
}))


# ---- Tabs ----
tab1, tab2, tab3 = st.tabs(["📈 Scatter Plots", "📊 Brain-PAD Histogram", "📦 Boxplot Comparison"])

# --- Tab 1: Scatter Plots ---
with tab1:
    st.markdown("### Brain Age vs Chronological Age by Model")

    # Option to choose whether to overlay or separate the correlation plots
    show_separate = st.radio("Choose to show correlations:", ("Separate Plots", "Overlay All Models"))

    # Colors for different models
    colors = px.colors.qualitative.Set2

    if show_separate == "Overlay All Models":
        fig = go.Figure()

        # Add scatter plots for all selected models in one plot
        for i, model in enumerate(selected_models):
            fig.add_trace(
                go.Scatter(
                    x=df_filtered["chronological_age"],
                    y=df_filtered[model],
                    mode="markers",
                    name=model,
                    marker=dict(color=colors[i], opacity=0.6),
                    hovertemplate=(
                        "Participant ID: %{customdata[0]}<br>"
                        "Chronological Age: %{x}<br>"
                        "Predicted Brain Age: %{y}"
                    ),
                    customdata=df_filtered[["participant_id"]],
                )
            )

        # Layout updates
        fig.update_layout(
            title="Correlation Plot for All Models",
            xaxis_title="Chronological Age",
            yaxis_title="Predicted Brain Age",
            template="plotly_white",
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        # Show separate plots for each selected model
        for i, model in enumerate(selected_models):
            st.markdown(f"**{model}**")
            fig = px.scatter(
                df_filtered,
                x="chronological_age",
                y=model,
                color="sex",
                trendline="ols",
                labels={
                    "chronological_age": "Chronological Age",
                    model: "Predicted Brain Age"
                },
                hover_data=["participant_id"],
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: Histograms ---
with tab2:
    st.markdown("### Brain-PAD Distribution by Model")

    # Colors for the histogram overlays
    colors = px.colors.qualitative.Set2

    # Overlaid histogram for each selected model
    fig = go.Figure()

    for i, model in enumerate(selected_models):
        brain_pad_col = f"brain_PAD_{model}"
        fig.add_trace(
            go.Histogram(
                x=df_filtered[brain_pad_col],
                name=model,
                opacity=0.6,
                nbinsx=30,
                marker=dict(color=colors[i]),
                histnorm="probability density",
                bingroup=1,
            )
        )

    fig.update_layout(
        title="Overlayed Brain-PAD Histograms for Selected Models",
        xaxis_title="Brain-PAD (Brain Age - Chronological Age)",
        yaxis_title="Density",
        template="plotly_white",
        barmode="overlay",
        legend_title="Models"
    )

    st.plotly_chart(fig, use_container_width=True)

# --- Tab 3: Boxplot with Individual Data Points Overlay ---
with tab3:
    st.markdown("### Brain-PAD by Model and Sex (Boxplot with Individual Data Points)")

    # Create a dataframe for melted data with Brain-PAD and Model
    melted = pd.melt(
        df_filtered,
        id_vars=["participant_id", "sex"],
        value_vars=[f"brain_PAD_{model}" for model in selected_models],
        var_name="Model",
        value_name="Brain-PAD"
    )

    # Remove the "brain_PAD_" prefix from the model names for readability
    melted["Model"] = melted["Model"].str.replace("brain_PAD_", "")

    # Create boxplots with individual data points overlay
    fig = px.box(
        melted,
        x="Model",
        y="Brain-PAD",
        color="sex",
        points="all",  # Display all individual points on top of the boxplot
        notched=True,
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    st.plotly_chart(fig, use_container_width=True)

