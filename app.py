
import streamlit as st
import pandas as pd

# --------------------------------
# PAGE SETTINGS
# --------------------------------

st.set_page_config(
    page_title="Titanic Passenger Insights",
    layout="wide"
)

# --------------------------------
# TASK 1: LOAD AND INSPECT DATA
# --------------------------------

df = pd.read_csv("titanic.csv")

st.title("Titanic Passenger Insights")

# --------------------------------
# TASK 2: HANDLE MISSING VALUES
# --------------------------------

cleaned_df = df.copy()

age_median = cleaned_df["age"].median()
cleaned_df["age"] = cleaned_df["age"].fillna(age_median)

embarked_mode = cleaned_df["embarked"].mode()[0]
cleaned_df["embarked"] = cleaned_df["embarked"].fillna(embarked_mode)

# --------------------------------
# TASK 3: CREATE TRANSFORMED FEATURES
# --------------------------------

transformed_df = cleaned_df.copy()

transformed_df["age_norm"] = (
    (transformed_df["age"] - transformed_df["age"].min()) /
    (transformed_df["age"].max() - transformed_df["age"].min())
)

transformed_df["fare_norm"] = (
    (transformed_df["fare"] - transformed_df["fare"].min()) /
    (transformed_df["fare"].max() - transformed_df["fare"].min())
)


def categorize_age(age):
    if age <= 12:
        return "Child"
    elif age <= 18:
        return "Teen"
    elif age <= 35:
        return "Young Adult"
    elif age <= 60:
        return "Adult"
    else:
        return "Older Adult"


transformed_df["age_group"] = transformed_df["age"].apply(categorize_age)

# --------------------------------
# TASK 4: DUMMY CODING
# --------------------------------

dummy_df = pd.get_dummies(
    transformed_df,
    columns=["sex", "embarked"],
    dtype=int
)

# --------------------------------
# TASK 5: SIDEBAR FILTERS
# --------------------------------

st.sidebar.header("Filters")

class_options = sorted(transformed_df["pclass"].unique().tolist())
sex_options = sorted(transformed_df["sex"].unique().tolist())

age_group_options = [
    "Child",
    "Teen",
    "Young Adult",
    "Adult",
    "Older Adult"
]

selected_class = st.sidebar.multiselect(
    "Passenger Class",
    options=class_options,
    default=class_options
)

selected_sex = st.sidebar.multiselect(
    "Sex",
    options=sex_options,
    default=sex_options
)

selected_age = st.sidebar.multiselect(
    "Age Group",
    options=age_group_options,
    default=age_group_options
)

filtered_df = transformed_df[
    (transformed_df["pclass"].isin(selected_class)) &
    (transformed_df["sex"].isin(selected_sex)) &
    (transformed_df["age_group"].isin(selected_age))
]

# --------------------------------
# FOUR TABS
# --------------------------------

overview_tab, explore_tab, preprocessing_tab, data_tab = st.tabs(
    ["Overview", "Explore", "Preprocessing", "Data Explorer"]
)

# --------------------------------
# OVERVIEW
# --------------------------------

with overview_tab:

    st.header("Overview")

    col1, col2, col3, col4 = st.columns(4)

    passenger_count = len(filtered_df)

    survivor_count = int(filtered_df["survived"].sum())

    survival_rate = (
        filtered_df["survived"].mean() * 100
        if passenger_count > 0 else 0
    )

    average_fare = (
        filtered_df["fare"].mean()
        if passenger_count > 0 else 0
    )

    col1.metric("Passengers", passenger_count)
    col2.metric("Survivors", survivor_count)
    col3.metric("Survival Rate", f"{survival_rate:.1f}%")
    col4.metric("Average Fare", f"{average_fare:.2f}")

    st.subheader("Survival Counts")

    survival_counts = (
        filtered_df["survived"]
        .map({0: "Did Not Survive", 1: "Survived"})
        .value_counts()
        .reindex(["Did Not Survive", "Survived"], fill_value=0)
    )

    st.bar_chart(survival_counts)

    st.subheader("Survival Rate by Passenger Class")

    survival_by_class = (
        filtered_df.groupby("pclass")["survived"].mean() * 100
    )

    st.bar_chart(survival_by_class)

    st.caption("Survival rates describe this dataset only.")

# --------------------------------
# EXPLORE
# --------------------------------

with explore_tab:

    st.header("Explore")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Survival Rate by Passenger Class")

        survival_by_class = (
            filtered_df.groupby("pclass")["survived"].mean() * 100
        )

        st.bar_chart(survival_by_class)

    with col2:

        st.subheader("Survival Rate by Sex")

        survival_by_sex = (
            filtered_df.groupby("sex")["survived"].mean() * 100
        )

        st.bar_chart(survival_by_sex)

    st.subheader("Age Distribution")

    age_histogram = pd.cut(
        filtered_df["age"],
        bins=[0, 10, 20, 30, 40, 50, 60, 70, 80],
        include_lowest=True
    ).value_counts().sort_index()
    
    age_histogram.index = age_histogram.index.astype(str)

    st.bar_chart(age_histogram)

    # TASK 6: ONE FINDING

    st.subheader("One Finding")

    class_survival = (
        df.groupby("pclass")["survived"].mean() * 100
    )

    highest_class = class_survival.idxmax()
    highest_rate = class_survival.max()

    st.write(
        f"In this dataset, passengers in class {highest_class} "
        f"had the highest survival rate ({highest_rate:.1f}%). "
        "This is a descriptive finding and does not show causation."
    )

# --------------------------------
# PREPROCESSING
# --------------------------------

with preprocessing_tab:

    st.header("Preprocessing")

    st.subheader("Missing Values Before Cleaning")

    st.write(df[["age", "embarked"]].isnull().sum())

    st.subheader("Missing Values After Cleaning")

    st.write(cleaned_df[["age", "embarked"]].isnull().sum())

    st.subheader("Normalized Age and Fare")

    st.dataframe(
        transformed_df[
            ["age", "age_norm", "fare", "fare_norm"]
        ].head(10)
    )

    st.subheader("Age Groups")

    st.dataframe(
        transformed_df[["age", "age_group"]].head(10)
    )

    st.subheader("Before Dummy Coding")

    st.dataframe(
        transformed_df[["sex", "embarked"]].head(10)
    )

    st.subheader("After Dummy Coding")

    st.dataframe(dummy_df.head(10))

# --------------------------------
# DATA EXPLORER
# --------------------------------

with data_tab:

    st.header("Data Explorer")

    dataset_choice = st.selectbox(
        "Choose Dataset",
        ["Raw", "Cleaned", "Transformed"]
    )

    if dataset_choice == "Raw":
        selected_df = df

    elif dataset_choice == "Cleaned":
        selected_df = cleaned_df

    else:
        selected_df = transformed_df

    st.write("Number of Rows:", selected_df.shape[0])

    st.write("Number of Columns:", selected_df.shape[1])

    selected_columns = st.multiselect(
        "Choose Columns",
        options=selected_df.columns.tolist(),
        default=selected_df.columns.tolist()
    )

    st.dataframe(
        selected_df[selected_columns],
        use_container_width=True
    )

    st.subheader("Column Names")

    st.write(selected_df.columns.tolist())

    st.subheader("Data Types")

    st.write(selected_df.dtypes)

    st.subheader("Missing Values")

    st.write(selected_df.isnull().sum())

    st.subheader("First Five Rows")

    st.dataframe(selected_df.iloc[0:5])
