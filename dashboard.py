


import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Simulate data for 200 participants
n = 200
participant_ids = [f"{i:03d}" for i in range(1, n + 1)]
chronological_ages = np.random.randint(18, 90, size=n)

# Simulate predicted brain age with some noise and bias
predicted_brain_ages = chronological_ages + np.random.normal(loc=0, scale=5, size=n)

# Assign random sexes
sexes = np.random.choice(['M', 'F'], size=n)

# Create the DataFrame
df_simulated = pd.DataFrame({
    "participant_id": participant_ids,
    "chronological_age": chronological_ages,
    "predicted_brain_age": predicted_brain_ages,
    "sex": sexes
})

# Save to CSV
csv_path = "your_brain_age_data.csv"
df_simulated.to_csv(csv_path, index=False)

csv_path


# generate interactive plot (simple)

import pandas as pd
import plotly.express as px

# Load your data
df = pd.read_csv("your_brain_age_data.csv")

# Calculate Brain-PAD (Predicted Age - Chronological Age)
df["brain_PAD"] = df["predicted_brain_age"] - df["chronological_age"]

# Simple scatter plot with optional color by group
fig = px.scatter(
    df,
    x="chronological_age",
    y="predicted_brain_age",
    color="sex",  # or diagnosis, or leave out for no color grouping
    hover_data=["participant_id", "brain_PAD"],
    labels={
        "chronological_age": "Chronological Age",
        "predicted_brain_age": "Predicted Brain Age",
        "sex": "Sex"
    },
    title="Predicted Brain Age vs. Chronological Age"
)

# Add diagonal reference line (perfect prediction)
fig.add_shape(
    type="line",
    x0=df["chronological_age"].min(),
    y0=df["chronological_age"].min(),
    x1=df["chronological_age"].max(),
    y1=df["chronological_age"].max(),
    line=dict(dash="dash", color="gray")
)

fig.show()



##Features to Add (Next Step Complexity):
#Filtering options (e.g., by sex, age range)
#Brain-PAD distribution plot (boxplot or histogram)
#Summary statistics panel (mean brain-PAD, correlation)
#(Optional) Group comparisons (e.g., mean Brain-PAD by sex)


conda create -n brainage-env -c conda-forge python=3.10 pyarrow streamlit plotly pandas
conda activate brainage-env


# Updated Streamlit Script (nano brain_age_dashboard.py)
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Load data
df = pd.read_csv("your_brain_age_data.csv")
df["brain_PAD"] = df["predicted_brain_age"] - df["chronological_age"]

st.title("🧠 Brain Age Dashboard")

# Sidebar filters
st.sidebar.header("Filter Participants")
sex_filter = st.sidebar.multiselect("Sex", options=df["sex"].unique(), default=df["sex"].unique())
age_range = st.sidebar.slider("Chronological Age Range", int(df["chronological_age"].min()), int(df["chronological_age"].max()), (20, 80))

# Apply filters
df_filtered = df[(df["sex"].isin(sex_filter)) &
                 (df["chronological_age"].between(age_range[0], age_range[1]))]

# Summary statistics
st.subheader("📊 Summary Statistics")
col1, col2, col3 = st.columns(3)
col1.metric("Avg Brain-PAD", f"{df_filtered['brain_PAD'].mean():.2f} years")
col2.metric("Std Dev Brain-PAD", f"{df_filtered['brain_PAD'].std():.2f}")
col3.metric("Age–Brain Age Corr.", f"{df_filtered[['chronological_age','predicted_brain_age']].corr().iloc[0,1]:.2f}")

# Main scatter plot
st.subheader("🔍 Predicted Brain Age vs. Chronological Age")
fig_scatter = px.scatter(
    df_filtered,
    x="chronological_age",
    y="predicted_brain_age",
    color="sex",
    hover_data=["participant_id", "brain_PAD"],
    labels={"chronological_age": "Chronological Age", "predicted_brain_age": "Predicted Brain Age"},
)
fig_scatter.add_shape(
    type="line",
    x0=df["chronological_age"].min(),
    y0=df["chronological_age"].min(),
    x1=df["chronological_age"].max(),
    y1=df["chronological_age"].max(),
    line=dict(dash="dash", color="gray")
)
st.plotly_chart(fig_scatter, use_container_width=True)

# Brain-PAD distribution
st.subheader("📈 Brain-PAD Distribution")
fig_hist = px.histogram(
    df_filtered,
    x="brain_PAD",
    color="sex",
    nbins=30,
    marginal="box",
    labels={"brain_PAD": "Brain-PAD (Predicted - Chronological Age)"},
)
st.plotly_chart(fig_hist, use_container_width=True)

# TO RUN
streamlit run brain_age_dashboard.py



# simulate data for 5 brain ages:
import pandas as pd
import numpy as np

# Simulate data for 5 different brain age models
np.random.seed(42)
n = 300

df_simulated = pd.DataFrame({
    "participant_id": [f"sub-{i}" for i in range(n)],
    "chronological_age": np.random.randint(20, 80, size=n),
    "sex": np.random.choice(["Male", "Female"], size=n)
})

# Simulate 5 predicted brain age models with slightly different characteristics
for i in range(1, 6):
    noise = np.random.normal(0, 5, size=n)
    bias = np.random.uniform(-2, 2)  # each model has a different bias
    df_simulated[f"predicted_brain_age_model{i}"] = df_simulated["chronological_age"] + bias + noise

# Add brain-PAD columns
for i in range(1, 6):
    df_simulated[f"brain_PAD_model{i}"] = df_simulated[f"predicted_brain_age_model{i}"] - df_simulated["chronological_age"]

df_simulated.head()
#  participant_id  chronological_age     sex  ...  brain_PAD_model3  brain_PAD_model4  brain_PAD_model5
#0          sub-0                 58    Male  ...         -2.845335         -2.040563          2.024803
#1          sub-1                 71    Male  ...         -3.252289         -0.662403          0.544988
#2          sub-2                 48  Female  ...          2.519217         -2.155644         -0.078429
#3          sub-3                 34    Male  ...         -3.372019         -1.212390         -0.036036
#4          sub-4                 62  Female  ...         -2.737512         -6.771660          4.456877


# Save to CSV
df_simulated.to_csv("five_brain_age_data.csv", index=False)


# to brain_age_dashboard2.py and the below
#nano brain_age_dashboard2.py

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# ---- Page config ----
st.set_page_config(page_title="Brain Age Dashboard", layout="wide")

st.title("🧠 Brain Age Dashboard")
st.markdown("Explore predicted brain ages, Brain-PAD, and age-related trends from multiple models.")

REQUIRED_COLUMNS = ["participant_id", "chronological_age", "sex"] + [f"model_{i}" for i in range(1, 6)]

@st.cache_data

def load_default_data():
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "participant_id": [f"subj_{i}" for i in range(n)],
        "chronological_age": np.random.randint(18, 85, n),
        "sex": np.random.choice(["Male", "Female"], n)
    })
    for i in range(1, 6):
        noise = np.random.normal(0, 5, n)
        bias = (i - 3) * 0.5  # slight bias per model
        df[f"model_{i}"] = df["chronological_age"] + noise + bias
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

# Compute Brain-PADs
for i in range(1, 6):
    df[f"brain_PAD_model_{i}"] = df[f"model_{i}"] - df["chronological_age"]

# ---- Sidebar filters ----
with st.sidebar:
    st.header("🔍 Filter Options")
    selected_sex = st.multiselect("Sex", df["sex"].unique(), default=list(df["sex"].unique()))
    age_min, age_max = st.slider("Chronological Age Range", 
                                 int(df["chronological_age"].min()), 
                                 int(df["chronological_age"].max()), 
                                 (20, 80))
    selected_models = st.multiselect("Select Model(s) to View", [f"model_{i}" for i in range(1, 6)], default=["model_1"])

df_filtered = df[
    (df["sex"].isin(selected_sex)) &
    (df["chronological_age"].between(age_min, age_max))
]

# ---- Summary Stats ----
st.subheader("📊 Key Statistics")
stat_cols = st.columns(len(selected_models))
for i, model in enumerate(selected_models):
    brain_pad_col = f"brain_PAD_{model}"
    mean_pad = df_filtered[brain_pad_col].mean()
    corr = df_filtered[["chronological_age", model]].corr().iloc[0, 1]
    stat_cols[i].metric(f"{model}: Avg Brain-PAD", f"{mean_pad:.2f} yrs")
    stat_cols[i].metric(f"{model}: Age Corr.", f"{corr:.2f}")

# ---- Tabs ----
tab1, tab2, tab3, tab4 = st.tabs(["Scatter Plot", "Histogram", "Boxplot", "Compare All Models"])

# --- Tab 1: Scatter
with tab1:
    st.markdown("### Predicted vs Chronological Age")
    for model in selected_models:
        fig = px.scatter(
            df_filtered,
            x="chronological_age",
            y=model,
            color="sex",
            trendline="ols",
            hover_data=["participant_id", f"brain_PAD_{model}"],
            labels={model: f"Predicted Brain Age ({model})"},
            template="plotly_white"
        )
        fig.add_shape(type="line",
                      x0=df_filtered["chronological_age"].min(),
                      y0=df_filtered["chronological_age"].min(),
                      x1=df_filtered["chronological_age"].max(),
                      y1=df_filtered["chronological_age"].max(),
                      line=dict(dash="dash", color="gray"))
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: Histogram
with tab2:
    st.markdown("### Brain-PAD Distribution")
    for model in selected_models:
        fig = px.histogram(
            df_filtered,
            x=f"brain_PAD_{model}",
            color="sex",
            nbins=30,
            marginal="box",
            opacity=0.75,
            labels={f"brain_PAD_{model}": f"Brain-PAD ({model})"},
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 3: Boxplot
with tab3:
    st.markdown("### Brain-PAD by Sex")
    for model in selected_models:
        fig = px.box(
            df_filtered,
            x="sex",
            y=f"brain_PAD_{model}",
            color="sex",
            points="all",
            notched=True,
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(
            yaxis_title=f"Brain-PAD ({model})",
            xaxis_title="Sex",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 4: All models side-by-side
with tab4:
    st.markdown("### Side-by-Side Comparison Across All Models")
    cols = st.columns(5)
    for i in range(1, 6):
        fig = px.histogram(
            df_filtered,
            x=f"brain_PAD_model_{i}",
            nbins=25,
            title=f"Model {i}",
            template="plotly_white",
            color_discrete_sequence=["#636EFA"]
        )
        fig.update_layout(margin=dict(t=30, b=10), title_x=0.5)
        cols[i - 1].plotly_chart(fig, use_container_width=True)



# RUN IT
streamlit run brain_age_dashboard2.py





