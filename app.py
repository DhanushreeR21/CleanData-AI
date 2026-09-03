import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CleanData AI",
    page_icon="🧹",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "original_df" not in st.session_state:
    st.session_state.original_df = None

if "df" not in st.session_state:
    st.session_state.df = None

if "history" not in st.session_state:
    st.session_state.history = []

if "dataset_versions" not in st.session_state:
    st.session_state.dataset_versions = []

if "file_id" not in st.session_state:
    st.session_state.file_id = None

if "auto_report" not in st.session_state:
    st.session_state.auto_report = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_quality_score(df):
    """Calculate basic dataset quality score."""

    if df is None or df.empty:
        return 0

    total_cells = df.shape[0] * df.shape[1]

    if total_cells == 0:
        return 0

    missing_percentage = (
        df.isnull().sum().sum() / total_cells
    ) * 100

    duplicate_percentage = (
        df.duplicated().sum() / max(len(df), 1)
    ) * 100

    empty_columns = sum(
        df[col].isnull().all() for col in df.columns
    )

    empty_column_penalty = (
        empty_columns / max(len(df.columns), 1)
    ) * 10

    score = (
        100
        - missing_percentage
        - duplicate_percentage
        - empty_column_penalty
    )

    return round(max(0, min(100, score)), 2)


def count_outliers(df):
    """Count IQR outliers across numeric columns."""

    if df is None or df.empty:
        return 0

    total_outliers = 0

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_columns:

        series = df[col].dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        total_outliers += (
            (series < lower) |
            (series > upper)
        ).sum()

    return int(total_outliers)


def save_history(action, dataframe):
    """Save current dataset version."""

    st.session_state.history.append(action)
    st.session_state.dataset_versions.append(
        dataframe.copy()
    )


def undo_last_action():

    if len(st.session_state.dataset_versions) > 1:

        st.session_state.dataset_versions.pop()

        st.session_state.history.pop()

        st.session_state.df = (
            st.session_state.dataset_versions[-1].copy()
        )

        return True

    return False


def clean_text_column(series, operation):

    if operation == "Strip Spaces":
        return series.astype(str).str.strip()

    elif operation == "Lowercase":
        return series.astype(str).str.lower()

    elif operation == "Uppercase":
        return series.astype(str).str.upper()

    elif operation == "Title Case":
        return series.astype(str).str.title()

    return series


def detect_column_quality(df):

    results = []

    for col in df.columns:

        missing = int(df[col].isnull().sum())

        missing_pct = round(
            (missing / len(df)) * 100, 2
        ) if len(df) > 0 else 0

        unique = df[col].nunique(
            dropna=True
        )

        dtype = str(df[col].dtype)

        if missing_pct == 0:
            status = "Good"
        elif missing_pct <= 20:
            status = "Warning"
        else:
            status = "Critical"

        results.append({
            "Column": col,
            "Data Type": dtype,
            "Missing Values": missing,
            "Missing %": missing_pct,
            "Unique Values": unique,
            "Quality": status
        })

    return pd.DataFrame(results)


def generate_quality_report(df):

    total_rows = len(df)
    total_columns = len(df.columns)

    missing_values = int(
        df.isnull().sum().sum()
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    outliers = count_outliers(df)

    empty_columns = [
        col for col in df.columns
        if df[col].isnull().all()
    ]

    constant_columns = [
        col for col in df.columns
        if df[col].nunique(dropna=False) <= 1
    ]

    quality_score = calculate_quality_score(df)

    report = {
        "Rows": total_rows,
        "Columns": total_columns,
        "Missing Values": missing_values,
        "Duplicate Rows": duplicate_rows,
        "Outliers": outliers,
        "Empty Columns": len(empty_columns),
        "Constant Columns": len(constant_columns),
        "Quality Score": quality_score
    }

    return report


def automatic_cleaning(
    df,
    remove_duplicates=True,
    fill_missing=True,
    clean_text=True,
    remove_empty_columns=True,
    remove_outliers=False
):

    cleaned = df.copy()

    actions = []

    before_rows = len(cleaned)
    before_columns = len(cleaned.columns)

    # --------------------------------------------------------
    # REMOVE EMPTY COLUMNS
    # --------------------------------------------------------

    if remove_empty_columns:

        empty_cols = [
            col for col in cleaned.columns
            if cleaned[col].isnull().all()
        ]

        if empty_cols:

            cleaned.drop(
                columns=empty_cols,
                inplace=True
            )

            actions.append(
                f"Removed {len(empty_cols)} empty column(s)"
            )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    if remove_duplicates:

        duplicate_count = int(
            cleaned.duplicated().sum()
        )

        if duplicate_count > 0:

            cleaned.drop_duplicates(
                inplace=True
            )

            actions.append(
                f"Removed {duplicate_count} duplicate row(s)"
            )

    # --------------------------------------------------------
    # CLEAN TEXT
    # --------------------------------------------------------

    if clean_text:

        text_columns = cleaned.select_dtypes(
            include=["object", "string"]
        ).columns

        text_count = 0

        for col in text_columns:

            original = cleaned[col].copy()

            cleaned[col] = (
                cleaned[col]
                .astype("string")
                .str.strip()
            )

            changed = (
                original.astype("string")
                != cleaned[col]
            ).sum()

            if changed > 0:
                text_count += 1

        if text_count > 0:

            actions.append(
                f"Cleaned {text_count} text column(s)"
            )

    # --------------------------------------------------------
    # HANDLE MISSING VALUES
    # --------------------------------------------------------

    if fill_missing:

        missing_before = int(
            cleaned.isnull().sum().sum()
        )

        for col in cleaned.columns:

            if cleaned[col].isnull().sum() == 0:
                continue

            if pd.api.types.is_numeric_dtype(
                cleaned[col]
            ):

                median_value = cleaned[col].median()

                if pd.notna(median_value):
                    cleaned[col] = cleaned[col].fillna(
                        median_value
                    )

            else:

                mode = cleaned[col].mode(
                    dropna=True
                )

                if not mode.empty:

                    cleaned[col] = cleaned[col].fillna(
                        mode.iloc[0]
                    )

                else:

                    cleaned[col] = cleaned[col].fillna(
                        "Unknown"
                    )

        missing_after = int(
            cleaned.isnull().sum().sum()
        )

        if missing_before > 0:

            actions.append(
                f"Handled {missing_before - missing_after} missing value(s)"
            )

    # --------------------------------------------------------
    # REMOVE OUTLIERS
    # --------------------------------------------------------

    if remove_outliers:

        numeric_columns = cleaned.select_dtypes(
            include=np.number
        ).columns

        if len(numeric_columns) > 0:

            original_length = len(cleaned)

            mask = pd.Series(
                True,
                index=cleaned.index
            )

            for col in numeric_columns:

                series = cleaned[col]

                if series.dropna().shape[0] < 4:
                    continue

                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)

                iqr = q3 - q1

                if iqr == 0:
                    continue

                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                mask &= (
                    (series >= lower) &
                    (series <= upper) |
                    series.isna()
                )

            cleaned = cleaned[mask].copy()

            removed = (
                original_length -
                len(cleaned)
            )

            if removed > 0:

                actions.append(
                    f"Removed {removed} outlier row(s)"
                )

    # --------------------------------------------------------
    # RESET INDEX
    # --------------------------------------------------------

    cleaned.reset_index(
        drop=True,
        inplace=True
    )

    after_rows = len(cleaned)
    after_columns = len(cleaned.columns)

    return (
        cleaned,
        actions,
        before_rows,
        after_rows,
        before_columns,
        after_columns
    )


def dataframe_to_excel(df):

    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="xlsxwriter"
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="Cleaned_Data"
        )

    return output.getvalue()


# ============================================================
# TITLE
# ============================================================

st.title("🧹 CleanData AI")

st.markdown(
    """
### Interactive Platform for Automated Data Cleaning and Visualization

Upload your dataset and clean, analyze, visualize and export
your data using an easy-to-use interface.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ CleanData AI")

menu = st.sidebar.radio(
    "Select Feature",
    [
        "Dataset Overview",
        "Missing Values",
        "Duplicates",
        "Data Types",
        "Delete Columns",
        "Outliers",
        "Rename Columns",
        "Text Cleaning",
        "Categorical Encoding",
        "Data Visualization",
        "🤖 AI Cleaning Suggestions",
        "⚡ Automated Cleaning Pipeline",
        "📊 Data Quality Dashboard",
        "📥 Export & Cleaning Report",
        "🔄 History & Undo"
    ]
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Dataset",
    type=["csv", "xlsx"]
)


if uploaded_file is not None:

    file_id = (
        uploaded_file.name +
        str(uploaded_file.size)
    )

    if st.session_state.file_id != file_id:

        try:

            if uploaded_file.name.lower().endswith(".csv"):

                df = pd.read_csv(uploaded_file)

            else:

                df = pd.read_excel(
                    uploaded_file
                )

            st.session_state.original_df = (
                df.copy()
            )

            st.session_state.df = (
                df.copy()
            )

            st.session_state.history = [
                "Dataset uploaded"
            ]

            st.session_state.dataset_versions = [
                df.copy()
            ]

            st.session_state.file_id = file_id

            st.session_state.auto_report = None

            st.success(
                "Dataset uploaded successfully!"
            )

        except Exception as e:

            st.error(
                f"Error reading dataset: {e}"
            )


# ============================================================
# CHECK DATASET
# ============================================================

if st.session_state.df is None:

    st.info(
        "👈 Please upload a CSV or Excel dataset from the sidebar."
    )

    st.stop()


df = st.session_state.df


# ============================================================
# RESET DATASET
# ============================================================

if st.sidebar.button(
    "🔄 Reset Dataset"
):

    st.session_state.df = (
        st.session_state.original_df.copy()
    )

    st.session_state.history = [
        "Dataset reset"
    ]

    st.session_state.dataset_versions = [
        st.session_state.original_df.copy()
    ]

    st.session_state.auto_report = None

    st.rerun()


# ============================================================
# CURRENT DATASET METRICS
# ============================================================

st.sidebar.markdown("---")

st.sidebar.metric(
    "Rows",
    df.shape[0]
)

st.sidebar.metric(
    "Columns",
    df.shape[1]
)

st.sidebar.metric(
    "Missing Values",
    int(df.isnull().sum().sum())
)

st.sidebar.metric(
    "Duplicates",
    int(df.duplicated().sum())
)

st.sidebar.metric(
    "Quality Score",
    f"{calculate_quality_score(df)}/100"
)


# ============================================================
# 1. DATASET OVERVIEW
# ============================================================

if menu == "Dataset Overview":

    st.header("📋 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Rows",
        df.shape[0]
    )

    col2.metric(
        "Columns",
        df.shape[1]
    )

    col3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    col4.metric(
        "Duplicate Rows",
        int(df.duplicated().sum())
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

    st.subheader("Dataset Information")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Non-Null Values": df.notnull().sum().values,
        "Missing Values": df.isnull().sum().values,
        "Unique Values": [
            df[col].nunique()
            for col in df.columns
        ]
    })

    st.dataframe(
        info_df,
        use_container_width=True
    )

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe(
            include="all"
        ).transpose(),
        use_container_width=True
    )


# ============================================================
# 2. MISSING VALUES
# ============================================================

elif menu == "Missing Values":

    st.header("🧩 Missing Value Handling")

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values,
        "Missing %": (
            df.isnull().mean() * 100
        ).round(2).values
    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )

    columns_with_missing = [
        col for col in df.columns
        if df[col].isnull().any()
    ]

    if not columns_with_missing:

        st.success(
            "✅ No missing values found!"
        )

    else:

        selected_column = st.selectbox(
            "Select Column",
            columns_with_missing
        )

        method = st.selectbox(
            "Select Method",
            [
                "Drop Rows",
                "Mean",
                "Median",
                "Mode",
                "Custom Value"
            ]
        )

        if method == "Custom Value":

            custom_value = st.text_input(
                "Enter replacement value"
            )

        if st.button(
            "Apply Missing Value Cleaning"
        ):

            new_df = df.copy()

            if method == "Drop Rows":

                new_df.dropna(
                    subset=[selected_column],
                    inplace=True
                )

            elif method == "Mean":

                if pd.api.types.is_numeric_dtype(
                    new_df[selected_column]
                ):

                    new_df[selected_column] = (
                        new_df[selected_column]
                        .fillna(
                            new_df[selected_column].mean()
                        )
                    )

                else:

                    st.error(
                        "Mean can only be used for numeric columns."
                    )

                    st.stop()

            elif method == "Median":

                if pd.api.types.is_numeric_dtype(
                    new_df[selected_column]
                ):

                    new_df[selected_column] = (
                        new_df[selected_column]
                        .fillna(
                            new_df[selected_column].median()
                        )
                    )

                else:

                    st.error(
                        "Median can only be used for numeric columns."
                    )

                    st.stop()

            elif method == "Mode":

                mode = (
                    new_df[selected_column]
                    .mode()
                )

                if not mode.empty:

                    new_df[selected_column] = (
                        new_df[selected_column]
                        .fillna(mode.iloc[0])
                    )

            elif method == "Custom Value":

                new_df[selected_column] = (
                    new_df[selected_column]
                    .fillna(custom_value)
                )

            st.session_state.df = new_df

            save_history(
                f"Missing values handled: {method}",
                new_df
            )

            st.success(
                "Missing values handled successfully!"
            )

            st.rerun()


# ============================================================
# 3. DUPLICATES
# ============================================================

elif menu == "Duplicates":

    st.header("♻️ Duplicate Rows")

    duplicate_count = int(
        df.duplicated().sum()
    )

    st.metric(
        "Duplicate Rows",
        duplicate_count
    )

    if duplicate_count > 0:

        st.dataframe(
            df[df.duplicated(keep=False)],
            use_container_width=True
        )

        if st.button(
            "🗑️ Remove Duplicate Rows"
        ):

            new_df = df.drop_duplicates().reset_index(
                drop=True
            )

            st.session_state.df = new_df

            save_history(
                "Duplicate rows removed",
                new_df
            )

            st.success(
                "Duplicate rows removed!"
            )

            st.rerun()

    else:

        st.success(
            "✅ No duplicate rows found."
        )


# ============================================================
# 4. DATA TYPES
# ============================================================

elif menu == "Data Types":

    st.header("🔤 Data Type Management")

    st.dataframe(
        pd.DataFrame({
            "Column": df.columns,
            "Current Type": df.dtypes.astype(str)
        }),
        use_container_width=True
    )

    selected_column = st.selectbox(
        "Select Column",
        df.columns
    )

    new_type = st.selectbox(
        "Convert To",
        [
            "int",
            "float",
            "string",
            "datetime"
        ]
    )

    if st.button(
        "Convert Data Type"
    ):

        new_df = df.copy()

        try:

            if new_type == "int":

                new_df[selected_column] = pd.to_numeric(
                    new_df[selected_column],
                    errors="coerce"
                ).astype("Int64")

            elif new_type == "float":

                new_df[selected_column] = pd.to_numeric(
                    new_df[selected_column],
                    errors="coerce"
                )

            elif new_type == "string":

                new_df[selected_column] = (
                    new_df[selected_column]
                    .astype("string")
                )

            elif new_type == "datetime":

                new_df[selected_column] = pd.to_datetime(
                    new_df[selected_column],
                    errors="coerce"
                )

            st.session_state.df = new_df

            save_history(
                f"Changed {selected_column} to {new_type}",
                new_df
            )

            st.success(
                "Data type changed successfully!"
            )

            st.rerun()

        except Exception as e:

            st.error(
                f"Conversion error: {e}"
            )


# ============================================================
# 5. DELETE COLUMNS
# ============================================================

elif menu == "Delete Columns":

    st.header("🗑️ Delete Columns")

    selected_columns = st.multiselect(
        "Select columns to delete",
        df.columns
    )

    if st.button(
        "Delete Selected Columns"
    ):

        if not selected_columns:

            st.warning(
                "Please select at least one column."
            )

        else:

            new_df = df.drop(
                columns=selected_columns
            )

            st.session_state.df = new_df

            save_history(
                f"Deleted columns: {', '.join(selected_columns)}",
                new_df
            )

            st.success(
                "Columns deleted successfully!"
            )

            st.rerun()


# ============================================================
# 6. OUTLIERS
# ============================================================

elif menu == "Outliers":

    st.header("📈 Outlier Detection")

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if not numeric_columns:

        st.warning(
            "No numeric columns available."
        )

    else:

        selected_column = st.selectbox(
            "Select Numeric Column",
            numeric_columns
        )

        q1 = df[selected_column].quantile(
            0.25
        )

        q3 = df[selected_column].quantile(
            0.75
        )

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_mask = (
            (df[selected_column] < lower) |
            (df[selected_column] > upper)
        )

        outliers = df[outlier_mask]

        st.metric(
            "Number of Outliers",
            len(outliers)
        )

        st.write(
            f"Lower Bound: **{lower:.2f}**"
        )

        st.write(
            f"Upper Bound: **{upper:.2f}**"
        )

        if len(outliers) > 0:

            st.dataframe(
                outliers,
                use_container_width=True
            )

            if st.button(
                "Remove Outliers"
            ):

                new_df = df[
                    ~outlier_mask
                ].reset_index(
                    drop=True
                )

                st.session_state.df = new_df

                save_history(
                    f"Removed outliers from {selected_column}",
                    new_df
                )

                st.success(
                    "Outliers removed!"
                )

                st.rerun()

        else:

            st.success(
                "✅ No outliers detected."
            )


# ============================================================
# 7. RENAME COLUMNS
# ============================================================

elif menu == "Rename Columns":

    st.header("✏️ Rename Columns")

    selected_column = st.selectbox(
        "Select Column",
        df.columns
    )

    new_name = st.text_input(
        "Enter New Column Name",
        value=selected_column
    )

    if st.button(
        "Rename Column"
    ):

        if not new_name.strip():

            st.error(
                "Column name cannot be empty."
            )

        elif (
            new_name != selected_column
            and new_name in df.columns
        ):

            st.error(
                "A column with this name already exists."
            )

        else:

            new_df = df.rename(
                columns={
                    selected_column:
                    new_name.strip()
                }
            )

            st.session_state.df = new_df

            save_history(
                f"Renamed {selected_column} to {new_name}",
                new_df
            )

            st.success(
                "Column renamed successfully!"
            )

            st.rerun()


# ============================================================
# 8. TEXT CLEANING
# ============================================================

elif menu == "Text Cleaning":

    st.header("🧹 Text Cleaning")

    text_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    if not text_columns:

        st.info(
            "No text columns found."
        )

    else:

        selected_column = st.selectbox(
            "Select Text Column",
            text_columns
        )

        operation = st.selectbox(
            "Cleaning Operation",
            [
                "Strip Spaces",
                "Lowercase",
                "Uppercase",
                "Title Case"
            ]
        )

        if st.button(
            "Clean Text"
        ):

            new_df = df.copy()

            new_df[selected_column] = (
                clean_text_column(
                    new_df[selected_column],
                    operation
                )
            )

            st.session_state.df = new_df

            save_history(
                f"{operation} applied to {selected_column}",
                new_df
            )

            st.success(
                "Text cleaning completed!"
            )

            st.rerun()


# ============================================================
# 9. CATEGORICAL ENCODING
# ============================================================

elif menu == "Categorical Encoding":

    st.header("🔢 Categorical Encoding")

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    if not categorical_columns:

        st.info(
            "No categorical columns found."
        )

    else:

        selected_column = st.selectbox(
            "Select Categorical Column",
            categorical_columns
        )

        encoding_method = st.selectbox(
            "Encoding Method",
            [
                "Label Encoding",
                "One-Hot Encoding"
            ]
        )

        if st.button(
            "Apply Encoding"
        ):

            new_df = df.copy()

            if encoding_method == "Label Encoding":

                new_df[selected_column] = pd.factorize(
                    new_df[selected_column]
                )[0]

            else:

                new_df = pd.get_dummies(
                    new_df,
                    columns=[selected_column],
                    dtype=int
                )

            st.session_state.df = new_df

            save_history(
                f"{encoding_method} applied to {selected_column}",
                new_df
            )

            st.success(
                "Encoding completed!"
            )

            st.rerun()


# ============================================================
# 10. DATA VISUALIZATION
# ============================================================

elif menu == "Data Visualization":

    st.header("📊 Interactive Data Visualization")

    chart_type = st.selectbox(
        "Select Chart",
        [
            "Histogram",
            "Bar Chart",
            "Scatter Plot",
            "Box Plot",
            "Pie Chart",
            "Correlation Heatmap"
        ]
    )

    if chart_type == "Histogram":

        column = st.selectbox(
            "Select Column",
            df.columns
        )

        fig = px.histogram(
            df,
            x=column,
            title=f"Distribution of {column}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    elif chart_type == "Bar Chart":

        column = st.selectbox(
            "Select Category Column",
            df.columns
        )

        counts = (
            df[column]
            .value_counts()
            .head(20)
            .reset_index()
        )

        counts.columns = [
            column,
            "Count"
        ]

        fig = px.bar(
            counts,
            x=column,
            y="Count",
            title=f"Bar Chart of {column}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    elif chart_type == "Scatter Plot":

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if len(numeric_columns) < 2:

            st.warning(
                "At least two numeric columns are required."
            )

        else:

            x_column = st.selectbox(
                "X Axis",
                numeric_columns
            )

            y_column = st.selectbox(
                "Y Axis",
                numeric_columns,
                index=1 if len(numeric_columns) > 1 else 0
            )

            fig = px.scatter(
                df,
                x=x_column,
                y=y_column,
                title=f"{y_column} vs {x_column}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    elif chart_type == "Box Plot":

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if not numeric_columns:

            st.warning(
                "No numeric columns found."
            )

        else:

            column = st.selectbox(
                "Select Numeric Column",
                numeric_columns
            )

            fig = px.box(
                df,
                y=column,
                title=f"Box Plot of {column}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    elif chart_type == "Pie Chart":

        column = st.selectbox(
            "Select Category Column",
            df.columns
        )

        counts = (
            df[column]
            .value_counts()
            .head(10)
            .reset_index()
        )

        counts.columns = [
            column,
            "Count"
        ]

        fig = px.pie(
            counts,
            names=column,
            values="Count",
            title=f"Distribution of {column}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    elif chart_type == "Correlation Heatmap":

        numeric_df = df.select_dtypes(
            include=np.number
        )

        if numeric_df.shape[1] < 2:

            st.warning(
                "At least two numeric columns are required."
            )

        else:

            corr = numeric_df.corr()

            fig = px.imshow(
                corr,
                text_auto=True,
                title="Correlation Heatmap"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# 11. AI CLEANING SUGGESTIONS
# ============================================================

elif menu == "🤖 AI Cleaning Suggestions":

    st.header("🤖 AI Cleaning Suggestions")

    st.write(
        "CleanData AI analyzes your dataset and recommends "
        "appropriate cleaning operations."
    )

    suggestions = []

    missing_count = int(
        df.isnull().sum().sum()
    )

    duplicate_count = int(
        df.duplicated().sum()
    )

    outlier_count = count_outliers(df)

    empty_columns = [
        col for col in df.columns
        if df[col].isnull().all()
    ]

    constant_columns = [
        col for col in df.columns
        if df[col].nunique(dropna=False) <= 1
    ]

    if missing_count > 0:

        suggestions.append(
            (
                "⚠️ Missing Values",
                f"{missing_count} missing values detected. "
                "Consider using median for numeric columns "
                "and mode for categorical columns."
            )
        )

    else:

        suggestions.append(
            (
                "✅ Missing Values",
                "No missing values detected."
            )
        )

    if duplicate_count > 0:

        suggestions.append(
            (
                "⚠️ Duplicate Rows",
                f"{duplicate_count} duplicate rows detected. "
                "Consider removing them."
            )
        )

    else:

        suggestions.append(
            (
                "✅ Duplicate Rows",
                "No duplicate rows detected."
            )
        )

    if outlier_count > 0:

        suggestions.append(
            (
                "⚠️ Outliers",
                f"Approximately {outlier_count} numeric "
                "outlier values detected using IQR."
            )
        )

    else:

        suggestions.append(
            (
                "✅ Outliers",
                "No significant IQR outliers detected."
            )
        )

    if empty_columns:

        suggestions.append(
            (
                "🚨 Empty Columns",
                f"Empty columns detected: "
                f"{', '.join(empty_columns)}"
            )
        )

    if constant_columns:

        suggestions.append(
            (
                "⚠️ Constant Columns",
                f"Columns with only one value: "
                f"{', '.join(constant_columns)}"
            )
        )

    for title, message in suggestions:

        st.subheader(title)
        st.info(message)

    st.markdown("---")

    score = calculate_quality_score(df)

    if score >= 90:

        st.success(
            f"Excellent dataset quality: {score}/100"
        )

    elif score >= 70:

        st.warning(
            f"Moderate dataset quality: {score}/100"
        )

    else:

        st.error(
            f"Poor dataset quality: {score}/100. "
            "Cleaning is recommended."
        )


# ============================================================
# 12. AUTOMATED CLEANING PIPELINE
# ============================================================

elif menu == "⚡ Automated Cleaning Pipeline":

    st.header(
        "⚡ Automated Data Cleaning Pipeline"
    )

    st.write(
        """
        Select the operations you want CleanData AI to perform.
        The system will clean the dataset automatically and show
        before/after results.
        """
    )

    st.subheader("Select Cleaning Operations")

    remove_duplicates = st.checkbox(
        "♻️ Remove duplicate rows",
        value=True
    )

    fill_missing = st.checkbox(
        "🧩 Handle missing values automatically",
        value=True
    )

    clean_text = st.checkbox(
        "🧹 Clean text columns",
        value=True
    )

    remove_empty_columns = st.checkbox(
        "🗑️ Remove completely empty columns",
        value=True
    )

    remove_outliers = st.checkbox(
        "📈 Remove numeric outlier rows",
        value=False
    )

    st.markdown("---")

    if st.button(
        "🚀 Run Automated Cleaning",
        type="primary"
    ):

        before_report = generate_quality_report(
            df
        )

        (
            cleaned_df,
            actions,
            before_rows,
            after_rows,
            before_columns,
            after_columns
        ) = automatic_cleaning(
            df,
            remove_duplicates,
            fill_missing,
            clean_text,
            remove_empty_columns,
            remove_outliers
        )

        after_report = generate_quality_report(
            cleaned_df
        )

        st.session_state.df = cleaned_df

        save_history(
            "Automated Cleaning Pipeline",
            cleaned_df
        )

        st.session_state.auto_report = {
            "before": before_report,
            "after": after_report,
            "actions": actions
        }

        st.success(
            "🎉 Automated cleaning completed successfully!"
        )

        if actions:

            st.subheader(
                "Cleaning Operations Performed"
            )

            for action in actions:

                st.write(
                    f"✅ {action}"
                )

        else:

            st.info(
                "No changes were required."
            )

        st.markdown("---")

        st.subheader(
            "Before vs After"
        )

        c1, c2 = st.columns(2)

        with c1:

            st.markdown(
                "### Before Cleaning"
            )

            st.metric(
                "Rows",
                before_report["Rows"]
            )

            st.metric(
                "Columns",
                before_report["Columns"]
            )

            st.metric(
                "Missing Values",
                before_report["Missing Values"]
            )

            st.metric(
                "Duplicates",
                before_report["Duplicate Rows"]
            )

            st.metric(
                "Quality Score",
                f'{before_report["Quality Score"]}/100'
            )

        with c2:

            st.markdown(
                "### After Cleaning"
            )

            st.metric(
                "Rows",
                after_report["Rows"]
            )

            st.metric(
                "Columns",
                after_report["Columns"]
            )

            st.metric(
                "Missing Values",
                after_report["Missing Values"]
            )

            st.metric(
                "Duplicates",
                after_report["Duplicate Rows"]
            )

            st.metric(
                "Quality Score",
                f'{after_report["Quality Score"]}/100'
            )

        st.markdown("---")

        st.subheader(
            "Cleaned Dataset Preview"
        )

        st.dataframe(
            cleaned_df.head(100),
            use_container_width=True
        )


# ============================================================
# 13. DATA QUALITY DASHBOARD
# ============================================================

elif menu == "📊 Data Quality Dashboard":

    st.header(
        "📊 Data Quality Dashboard"
    )

    report = generate_quality_report(
        df
    )

    score = report["Quality Score"]

    if score >= 90:

        st.success(
            f"🟢 Excellent Data Quality — {score}/100"
        )

    elif score >= 70:

        st.warning(
            f"🟡 Moderate Data Quality — {score}/100"
        )

    else:

        st.error(
            f"🔴 Poor Data Quality — {score}/100"
        )

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Rows",
        report["Rows"]
    )

    c2.metric(
        "Columns",
        report["Columns"]
    )

    c3.metric(
        "Missing Values",
        report["Missing Values"]
    )

    c4.metric(
        "Duplicate Rows",
        report["Duplicate Rows"]
    )

    c5, c6, c7, c8 = st.columns(4)

    c5.metric(
        "Outliers",
        report["Outliers"]
    )

    c6.metric(
        "Empty Columns",
        report["Empty Columns"]
    )

    c7.metric(
        "Constant Columns",
        report["Constant Columns"]
    )

    c8.metric(
        "Quality Score",
        f'{report["Quality Score"]}/100'
    )

    st.markdown("---")

    st.subheader(
        "📋 Column-Level Quality"
    )

    quality_df = detect_column_quality(
        df
    )

    st.dataframe(
        quality_df,
        use_container_width=True
    )

    st.subheader(
        "📊 Missing Values by Column"
    )

    missing_chart = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values
    })

    missing_chart = missing_chart[
        missing_chart["Missing Values"] > 0
    ]

    if not missing_chart.empty:

        fig = px.bar(
            missing_chart,
            x="Column",
            y="Missing Values",
            title="Missing Values by Column"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.success(
            "No missing values found."
        )

    st.subheader(
        "📈 Outlier Summary"
    )

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    outlier_summary = []

    for col in numeric_columns:

        series = df[col].dropna()

        if len(series) < 4:

            outlier_summary.append({
                "Column": col,
                "Outliers": 0
            })

            continue

        q1 = series.quantile(
            0.25
        )

        q3 = series.quantile(
            0.75
        )

        iqr = q3 - q1

        if iqr == 0:

            count = 0

        else:

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            count = int(
                (
                    (series < lower) |
                    (series > upper)
                ).sum()
            )

        outlier_summary.append({
            "Column": col,
            "Outliers": count
        })

    if outlier_summary:

        outlier_df = pd.DataFrame(
            outlier_summary
        )

        st.dataframe(
            outlier_df,
            use_container_width=True
        )


# ============================================================
# 14. EXPORT & CLEANING REPORT
# ============================================================

elif menu == "📥 Export & Cleaning Report":

    st.header(
        "📥 Export & Cleaning Report"
    )

    report = generate_quality_report(
        df
    )

    st.subheader(
        "Current Dataset Quality"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Rows",
        report["Rows"]
    )

    c2.metric(
        "Columns",
        report["Columns"]
    )

    c3.metric(
        "Missing Values",
        report["Missing Values"]
    )

    c4.metric(
        "Quality Score",
        f'{report["Quality Score"]}/100'
    )

    st.markdown("---")

    st.subheader(
        "📄 Cleaning Report"
    )

    report_text = f"""
CleanData AI - Data Cleaning Report
====================================

Dataset Rows: {report["Rows"]}
Dataset Columns: {report["Columns"]}

Missing Values: {report["Missing Values"]}
Duplicate Rows: {report["Duplicate Rows"]}
Outliers: {report["Outliers"]}
Empty Columns: {report["Empty Columns"]}
Constant Columns: {report["Constant Columns"]}

Overall Quality Score:
{report["Quality Score"]}/100

Cleaning History:
"""

    for i, action in enumerate(
        st.session_state.history,
        start=1
    ):

        report_text += (
            f"\n{i}. {action}"
        )

    st.text_area(
        "Report Preview",
        report_text,
        height=350
    )

    st.download_button(
        "📄 Download Cleaning Report",
        data=report_text,
        file_name="CleanData_AI_Report.txt",
        mime="text/plain"
    )

    st.markdown("---")

    st.subheader(
        "💾 Download Cleaned Dataset"
    )

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download CSV",
        data=csv_data,
        file_name="Cleaned_Dataset.csv",
        mime="text/csv"
    )

    excel_data = dataframe_to_excel(
        df
    )

    st.download_button(
        "⬇️ Download Excel",
        data=excel_data,
        file_name="Cleaned_Dataset.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ============================================================
# 15. HISTORY & UNDO
# ============================================================

elif menu == "🔄 History & Undo":

    st.header(
        "🔄 Cleaning History & Undo"
    )

    if not st.session_state.history:

        st.info(
            "No cleaning history available."
        )

    else:

        for i, action in enumerate(
            st.session_state.history,
            start=1
        ):

            st.write(
                f"**{i}.** {action}"
            )

    st.markdown("---")

    st.subheader(
        "Current Dataset Version"
    )

    st.write(
        f"Rows: **{len(df)}**"
    )

    st.write(
        f"Columns: **{len(df.columns)}**"
    )

    if len(
        st.session_state.dataset_versions
    ) > 1:

        if st.button(
            "↩️ Undo Last Cleaning Operation"
        ):

            if undo_last_action():

                st.success(
                    "Last operation undone successfully!"
                )

                st.rerun()

    else:

        st.info(
            "There is no operation available to undo."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🧹 CleanData AI | Automated Data Cleaning & Visualization Platform"
)