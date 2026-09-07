import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CleanData AI",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    color: #1557C0;
}

.subtitle {
    font-size: 22px;
    font-weight: 600;
    color: #263238;
}

.section-title {
    font-size: 28px;
    font-weight: 700;
    color: #1557C0;
}

.metric-card {
    background-color: #F5F9FF;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #D8E7FF;
    text-align: center;
}

.feature-card {
    padding: 22px;
    border-radius: 15px;
    background-color: #F8FBFF;
    border: 1px solid #DCEAFF;
    min-height: 180px;
}

.small-text {
    color: #5F6B7A;
    font-size: 16px;
    line-height: 1.6;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "original_df" not in st.session_state:
    st.session_state.original_df = None

if "history" not in st.session_state:
    st.session_state.history = []

if "dataset_versions" not in st.session_state:
    st.session_state.dataset_versions = []

if "file_id" not in st.session_state:
    st.session_state.file_id = None

if "auto_report" not in st.session_state:
    st.session_state.auto_report = None


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def calculate_quality_score(df):

    if df is None or df.empty:
        return 0

    total_cells = df.shape[0] * df.shape[1]

    if total_cells == 0:
        return 0

    missing = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    missing_ratio = missing / total_cells
    duplicate_ratio = duplicate_rows / max(len(df), 1)

    score = 100

    score -= missing_ratio * 50
    score -= duplicate_ratio * 30

    return max(0, min(100, round(score, 2)))


def count_outliers(df):

    if df is None or df.empty:
        return 0

    total_outliers = 0

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        series = df[column].dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        total_outliers += int(
            ((series < lower) | (series > upper)).sum()
        )

    return total_outliers


def save_history(action, dataframe):

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

    for column in df.columns:

        missing = int(df[column].isnull().sum())

        unique = int(df[column].nunique())

        dtype = str(df[column].dtype)

        results.append({
            "Column": column,
            "Data Type": dtype,
            "Missing Values": missing,
            "Unique Values": unique
        })

    return pd.DataFrame(results)


def generate_quality_report(df):

    empty_columns = [
        col for col in df.columns
        if df[col].isnull().all()
    ]

    constant_columns = [
        col for col in df.columns
        if df[col].nunique(dropna=False) <= 1
    ]

    report = {
        "Rows": len(df),
        "Columns": len(df.columns),
        "Missing Values": int(df.isnull().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum()),
        "Outliers": count_outliers(df),
        "Empty Columns": len(empty_columns),
        "Constant Columns": len(constant_columns),
        "Quality Score": calculate_quality_score(df)
    }

    return report


def automatic_cleaning(
    df,
    remove_duplicates=True,
    handle_missing=True,
    clean_text=True,
    remove_empty_columns=True,
    remove_outliers=False
):

    cleaned = df.copy()

    actions = []

    before_rows = len(cleaned)
    before_missing = int(cleaned.isnull().sum().sum())

    # -----------------------------------------
    # Remove duplicate rows
    # -----------------------------------------

    if remove_duplicates:

        duplicate_count = int(cleaned.duplicated().sum())

        if duplicate_count > 0:

            cleaned = cleaned.drop_duplicates()

            actions.append(
                f"Removed {duplicate_count} duplicate rows"
            )

    # -----------------------------------------
    # Handle missing values
    # -----------------------------------------

    if handle_missing:

        for column in cleaned.columns:

            if cleaned[column].isnull().sum() == 0:
                continue

            if pd.api.types.is_numeric_dtype(
                cleaned[column]
            ):

                median_value = cleaned[column].median()

                cleaned[column] = cleaned[column].fillna(
                    median_value
                )

                actions.append(
                    f"Filled missing values in '{column}' "
                    f"with median"
                )

            else:

                mode_values = cleaned[column].mode()

                if len(mode_values) > 0:

                    mode_value = mode_values.iloc[0]

                    cleaned[column] = cleaned[column].fillna(
                        mode_value
                    )

                    actions.append(
                        f"Filled missing values in '{column}' "
                        f"with mode"
                    )

                else:

                    cleaned[column] = cleaned[column].fillna(
                        "Unknown"
                    )

    # -----------------------------------------
    # Clean text columns
    # -----------------------------------------

    if clean_text:

        text_columns = cleaned.select_dtypes(
            include=["object", "string"]
        ).columns

        for column in text_columns:

            cleaned[column] = (
                cleaned[column]
                .astype(str)
                .str.strip()
            )

        if len(text_columns) > 0:

            actions.append(
                f"Trimmed spaces from {len(text_columns)} text columns"
            )

    # -----------------------------------------
    # Remove empty columns
    # -----------------------------------------

    if remove_empty_columns:

        empty_columns = [
            col for col in cleaned.columns
            if cleaned[col].isnull().all()
        ]

        if empty_columns:

            cleaned = cleaned.drop(
                columns=empty_columns
            )

            actions.append(
                f"Removed {len(empty_columns)} empty columns"
            )

    # -----------------------------------------
    # Remove numeric outliers
    # -----------------------------------------

    if remove_outliers:

        numeric_columns = cleaned.select_dtypes(
            include=np.number
        ).columns

        rows_to_keep = pd.Series(
            True,
            index=cleaned.index
        )

        removed_count = 0

        for column in numeric_columns:

            series = cleaned[column]

            if series.dropna().empty:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            condition = (
                (series >= lower) &
                (series <= upper)
            )

            rows_to_keep &= condition.fillna(True)

        removed_count = int(
            (~rows_to_keep).sum()
        )

        cleaned = cleaned.loc[rows_to_keep]

        if removed_count > 0:

            actions.append(
                f"Removed {removed_count} outlier rows"
            )

    after_rows = len(cleaned)
    after_missing = int(cleaned.isnull().sum().sum())

    report = {
        "Before Rows": before_rows,
        "After Rows": after_rows,
        "Before Missing Values": before_missing,
        "After Missing Values": after_missing,
        "Actions": actions
    }

    return cleaned, report


def dataframe_to_excel(df):

    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="xlsxwriter"
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="Cleaned Data"
        )

    return output.getvalue()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "<h1 style='color:#1557C0;'>🧹 CleanData AI</h1>",
        unsafe_allow_html=True
    )

    st.write(
        "Interactive Data Cleaning & Visualization"
    )

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📂 Upload Dataset",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:

        current_file_id = (
            uploaded_file.name,
            uploaded_file.size
        )

        if (
            st.session_state.file_id
            != current_file_id
        ):

            try:

                if uploaded_file.name.lower().endswith(
                    ".csv"
                ):

                    df = pd.read_csv(uploaded_file)

                else:

                    df = pd.read_excel(
                        uploaded_file
                    )

                st.session_state.df = df.copy()

                st.session_state.original_df = df.copy()

                st.session_state.history = [
                    "Initial Dataset"
                ]

                st.session_state.dataset_versions = [
                    df.copy()
                ]

                st.session_state.file_id = (
                    current_file_id
                )

                st.session_state.auto_report = None

                st.success(
                    "Dataset uploaded successfully!"
                )

            except Exception as e:

                st.error(
                    f"Error reading file: {e}"
                )

    st.markdown("---")

    menu = st.radio(
        "Navigation",
        [
            "🏠 Welcome",
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

    st.markdown("---")

    if st.session_state.df is not None:

        if st.button(
            "🔄 Reset Dataset",
            use_container_width=True
        ):

            st.session_state.df = (
                st.session_state.original_df.copy()
            )

            st.session_state.history = [
                "Initial Dataset"
            ]

            st.session_state.dataset_versions = [
                st.session_state.original_df.copy()
            ]

            st.session_state.auto_report = None

            st.success(
                "Dataset reset successfully!"
            )

            st.rerun()


# =========================================================
# WELCOME PAGE
# =========================================================

if menu == "🏠 Welcome":

    st.markdown(
        "<div class='main-title'>Welcome to CleanData AI</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>"
        "Intelligent Data Cleaning & Visualization Platform"
        "</div>",
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2 = st.columns(
        [1, 1.15]
    )

    with col1:

        st.markdown("""
        <div style="padding-top:35px;">

        <h2 style="color:#1557C0;">
        🧹 Clean Your Data Smarter
        </h2>

        <p class="small-text">
        CleanData AI helps you clean, analyze and visualize
        your data in a few clicks.
        </p>

        <p class="small-text">
        Upload your CSV or Excel dataset and use powerful
        data cleaning tools, AI-based suggestions,
        interactive visualizations and automated cleaning.
        </p>

        <h3 style="color:#1557C0;">
        Clean &nbsp; | &nbsp;
        Analyze &nbsp; | &nbsp;
        Visualize &nbsp; | &nbsp;
        Export
        </h3>

        </div>
        """, unsafe_allow_html=True)

        if st.session_state.df is None:

            st.info(
                "👈 Upload your CSV or Excel dataset "
                "from the sidebar to get started."
            )

    with col2:

        # Welcome image
        # Keep welcome.jpeg inside: assets/welcome.jpeg
        image_path = Path(__file__).parent / "assets" / "welcome.jpeg"

        if image_path.exists():

            st.image(
                str(image_path),
                use_container_width=True
            )

        else:

            st.warning(
                "Welcome image not found. "
                "Place welcome.jpeg inside the assets folder."
            )

    st.markdown("---")

    st.markdown(
        "<h2 style='text-align:center;'>"
        "✨ What Can CleanData AI Do?"
        "</h2>",
        unsafe_allow_html=True
    )

    st.write("")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("""
        <div class="feature-card">

        <h3 style="color:#1557C0;">
        🧹 Automated Cleaning
        </h3>

        <p class="small-text">
        Remove duplicates, handle missing values,
        fix data types and detect outliers easily.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class="feature-card">

        <h3 style="color:#159447;">
        🤖 AI Suggestions
        </h3>

        <p class="small-text">
        Get intelligent suggestions to improve
        your dataset quality and consistency.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown("""
        <div class="feature-card">

        <h3 style="color:#7B3FC6;">
        📊 Visualization
        </h3>

        <p class="small-text">
        Explore your data using interactive charts,
        graphs and correlation analysis.
        </p>

        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.markdown(
        "<h2 style='text-align:center;'>"
        "🚀 How It Works"
        "</h2>",
        unsafe_allow_html=True
    )

    h1, h2, h3, h4 = st.columns(4)

    with h1:

        st.markdown("### 1️⃣ Upload")
        st.write(
            "Upload your CSV or Excel dataset."
        )

    with h2:

        st.markdown("### 2️⃣ Clean")
        st.write(
            "Clean and preprocess your data."
        )

    with h3:

        st.markdown("### 3️⃣ Analyze")
        st.write(
            "Analyze quality and discover insights."
        )

    with h4:

        st.markdown("### 4️⃣ Export")
        st.write(
            "Download your cleaned dataset."
        )

    st.markdown("---")

    st.info(
        "💡 Start by uploading your dataset using "
        "the Upload Dataset option in the sidebar."
    )

    st.markdown(
        "<p style='text-align:center;color:#777;'>"
        "Built with ❤️ using Python & Streamlit"
        "</p>",
        unsafe_allow_html=True
    )


# =========================================================
# CHECK DATASET
# =========================================================

elif st.session_state.df is None:

    st.warning(
        "👈 Please upload a CSV or Excel dataset "
        "from the sidebar first."
    )

    st.stop()


df = st.session_state.df


# =========================================================
# DATASET OVERVIEW
# =========================================================

if menu == "Dataset Overview":

    st.markdown(
        "<div class='section-title'>📋 Dataset Overview</div>",
        unsafe_allow_html=True
    )

    st.write("")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Rows",
            df.shape[0]
        )

    with c2:
        st.metric(
            "Columns",
            df.shape[1]
        )

    with c3:
        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    with c4:
        st.metric(
            "Duplicate Rows",
            int(df.duplicated().sum())
        )

    st.markdown("---")

    st.subheader("📄 Dataset Preview")

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

    st.subheader("📊 Dataset Information")

    st.dataframe(
        detect_column_quality(df),
        use_container_width=True
    )


# =========================================================
# MISSING VALUES
# =========================================================

elif menu == "Missing Values":

    st.markdown(
        "<div class='section-title'>🔍 Missing Values</div>",
        unsafe_allow_html=True
    )

    missing_count = df.isnull().sum()

    missing_df = pd.DataFrame({
        "Column": missing_count.index,
        "Missing Values": missing_count.values
    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )

    columns_with_missing = [
        col for col in df.columns
        if df[col].isnull().sum() > 0
    ]

    if not columns_with_missing:

        st.success(
            "✅ No missing values found!"
        )

    else:

        selected_column = st.selectbox(
            "Select column",
            columns_with_missing
        )

        method = st.selectbox(
            "Select method",
            [
                "Drop Rows",
                "Mean",
                "Median",
                "Mode",
                "Custom Value"
            ]
        )

        custom_value = None

        if method == "Custom Value":

            custom_value = st.text_input(
                "Enter custom value"
            )

        if st.button(
            "Apply Missing Value Treatment",
            type="primary"
        ):

            new_df = df.copy()

            if method == "Drop Rows":

                new_df = new_df.dropna(
                    subset=[selected_column]
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

                mode = new_df[
                    selected_column
                ].mode()

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
                f"Missing values: {method} - {selected_column}",
                new_df
            )

            st.success(
                "Missing values handled successfully!"
            )

            st.rerun()


# =========================================================
# DUPLICATES
# =========================================================

elif menu == "Duplicates":

    st.markdown(
        "<div class='section-title'>♻️ Duplicate Detection</div>",
        unsafe_allow_html=True
    )

    duplicate_count = int(
        df.duplicated().sum()
    )

    st.metric(
        "Duplicate Rows",
        duplicate_count
    )

    if duplicate_count > 0:

        st.dataframe(
            df[df.duplicated()],
            use_container_width=True
        )

        if st.button(
            "🗑️ Remove Duplicate Rows",
            type="primary"
        ):

            new_df = df.drop_duplicates()

            st.session_state.df = new_df

            save_history(
                "Removed duplicate rows",
                new_df
            )

            st.success(
                "Duplicate rows removed!"
            )

            st.rerun()

    else:

        st.success(
            "✅ No duplicate rows found!"
        )


# =========================================================
# DATA TYPES
# =========================================================

elif menu == "Data Types":

    st.markdown(
        "<div class='section-title'>🔤 Data Type Conversion</div>",
        unsafe_allow_html=True
    )

    column = st.selectbox(
        "Select column",
        df.columns
    )

    data_type = st.selectbox(
        "Select new data type",
        [
            "int",
            "float",
            "string",
            "datetime"
        ]
    )

    if st.button(
        "Convert Data Type",
        type="primary"
    ):

        new_df = df.copy()

        try:

            if data_type == "int":

                new_df[column] = pd.to_numeric(
                    new_df[column],
                    errors="coerce"
                ).astype("Int64")

            elif data_type == "float":

                new_df[column] = pd.to_numeric(
                    new_df[column],
                    errors="coerce"
                )

            elif data_type == "string":

                new_df[column] = (
                    new_df[column].astype(str)
                )

            elif data_type == "datetime":

                new_df[column] = pd.to_datetime(
                    new_df[column],
                    errors="coerce"
                )

            st.session_state.df = new_df

            save_history(
                f"Converted {column} to {data_type}",
                new_df
            )

            st.success(
                "Data type converted successfully!"
            )

            st.rerun()

        except Exception as e:

            st.error(
                f"Conversion failed: {e}"
            )


# =========================================================
# DELETE COLUMNS
# =========================================================

elif menu == "Delete Columns":

    st.markdown(
        "<div class='section-title'>🗑️ Delete Columns</div>",
        unsafe_allow_html=True
    )

    selected_columns = st.multiselect(
        "Select columns to delete",
        df.columns
    )

    if st.button(
        "Delete Selected Columns",
        type="primary"
    ):

        if not selected_columns:

            st.warning(
                "Please select at least one column."
            )

        elif len(selected_columns) >= len(df.columns):

            st.error(
                "You cannot delete all columns."
            )

        else:

            new_df = df.drop(
                columns=selected_columns
            )

            st.session_state.df = new_df

            save_history(
                f"Deleted columns: {selected_columns}",
                new_df
            )

            st.success(
                "Columns deleted successfully!"
            )

            st.rerun()


# =========================================================
# OUTLIERS
# =========================================================

elif menu == "Outliers":

    st.markdown(
        "<div class='section-title'>📈 Outlier Detection</div>",
        unsafe_allow_html=True
    )

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if not numeric_columns:

        st.warning(
            "No numeric columns available."
        )

    else:

        column = st.selectbox(
            "Select numeric column",
            numeric_columns
        )

        series = df[column].dropna()

        if len(series) >= 4:

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outlier_mask = (
                (df[column] < lower) |
                (df[column] > upper)
            )

            outlier_count = int(
                outlier_mask.sum()
            )

            st.metric(
                "Detected Outliers",
                outlier_count
            )

            st.write(
                f"Lower Bound: **{lower:.2f}**"
            )

            st.write(
                f"Upper Bound: **{upper:.2f}**"
            )

            if outlier_count > 0:

                st.dataframe(
                    df[outlier_mask],
                    use_container_width=True
                )

                if st.button(
                    "🧹 Remove Outlier Rows",
                    type="primary"
                ):

                    new_df = df[
                        ~outlier_mask
                    ].copy()

                    st.session_state.df = new_df

                    save_history(
                        f"Removed outliers from {column}",
                        new_df
                    )

                    st.success(
                        "Outliers removed successfully!"
                    )

                    st.rerun()

            else:

                st.success(
                    "✅ No outliers detected."
                )


# =========================================================
# RENAME COLUMNS
# =========================================================

elif menu == "Rename Columns":

    st.markdown(
        "<div class='section-title'>✏️ Rename Columns</div>",
        unsafe_allow_html=True
    )

    column = st.selectbox(
        "Select column",
        df.columns
    )

    new_name = st.text_input(
        "Enter new column name",
        value=column
    )

    if st.button(
        "Rename Column",
        type="primary"
    ):

        if new_name.strip() == "":

            st.warning(
                "Column name cannot be empty."
            )

        elif (
            new_name != column
            and new_name in df.columns
        ):

            st.error(
                "A column with this name already exists."
            )

        else:

            new_df = df.rename(
                columns={
                    column: new_name
                }
            )

            st.session_state.df = new_df

            save_history(
                f"Renamed {column} to {new_name}",
                new_df
            )

            st.success(
                "Column renamed successfully!"
            )

            st.rerun()


# =========================================================
# TEXT CLEANING
# =========================================================

elif menu == "Text Cleaning":

    st.markdown(
        "<div class='section-title'>📝 Text Cleaning</div>",
        unsafe_allow_html=True
    )

    text_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    if not text_columns:

        st.warning(
            "No text columns found."
        )

    else:

        column = st.selectbox(
            "Select text column",
            text_columns
        )

        operation = st.selectbox(
            "Select cleaning operation",
            [
                "Strip Spaces",
                "Lowercase",
                "Uppercase",
                "Title Case"
            ]
        )

        if st.button(
            "Apply Text Cleaning",
            type="primary"
        ):

            new_df = df.copy()

            new_df[column] = clean_text_column(
                new_df[column],
                operation
            )

            st.session_state.df = new_df

            save_history(
                f"Text cleaning: {operation} - {column}",
                new_df
            )

            st.success(
                "Text cleaning applied successfully!"
            )

            st.rerun()


# =========================================================
# CATEGORICAL ENCODING
# =========================================================

elif menu == "Categorical Encoding":

    st.markdown(
        "<div class='section-title'>🔢 Categorical Encoding</div>",
        unsafe_allow_html=True
    )

    categorical_columns = df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    if not categorical_columns:

        st.warning(
            "No categorical columns found."
        )

    else:

        column = st.selectbox(
            "Select categorical column",
            categorical_columns
        )

        encoding_method = st.selectbox(
            "Select encoding method",
            [
                "Label Encoding",
                "One-Hot Encoding"
            ]
        )

        if st.button(
            "Apply Encoding",
            type="primary"
        ):

            new_df = df.copy()

            if encoding_method == "Label Encoding":

                new_df[column] = pd.factorize(
                    new_df[column]
                )[0]

                st.session_state.df = new_df

                save_history(
                    f"Label encoded {column}",
                    new_df
                )

            else:

                new_df = pd.get_dummies(
                    new_df,
                    columns=[column],
                    dtype=int
                )

                st.session_state.df = new_df

                save_history(
                    f"One-hot encoded {column}",
                    new_df
                )

            st.success(
                "Encoding applied successfully!"
            )

            st.rerun()


# =========================================================
# DATA VISUALIZATION
# =========================================================

elif menu == "Data Visualization":

    st.markdown(
        "<div class='section-title'>📊 Data Visualization</div>",
        unsafe_allow_html=True
    )

    chart_type = st.selectbox(
        "Select chart type",
        [
            "Histogram",
            "Bar Chart",
            "Scatter Plot",
            "Box Plot",
            "Pie Chart",
            "Correlation Heatmap"
        ]
    )

    # -----------------------------------------
    # Histogram
    # -----------------------------------------

    if chart_type == "Histogram":

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numeric_columns:

            column = st.selectbox(
                "Select column",
                numeric_columns
            )

            fig = px.histogram(
                df,
                x=column,
                title=f"Histogram of {column}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "No numeric columns available."
            )

    # -----------------------------------------
    # Bar Chart
    # -----------------------------------------

    elif chart_type == "Bar Chart":

        columns = df.columns.tolist()

        x_column = st.selectbox(
            "Select X-axis column",
            columns
        )

        y_numeric = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if y_numeric:

            y_column = st.selectbox(
                "Select Y-axis column",
                y_numeric
            )

            fig = px.bar(
                df,
                x=x_column,
                y=y_column,
                title=f"{y_column} by {x_column}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "No numeric columns available for Y-axis."
            )

    # -----------------------------------------
    # Scatter Plot
    # -----------------------------------------

    elif chart_type == "Scatter Plot":

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if len(numeric_columns) >= 2:

            x_column = st.selectbox(
                "Select X-axis",
                numeric_columns
            )

            y_column = st.selectbox(
                "Select Y-axis",
                numeric_columns,
                index=1
            )

            fig = px.scatter(
                df,
                x=x_column,
                y=y_column,
                title=f"{x_column} vs {y_column}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "At least two numeric columns are required."
            )

    # -----------------------------------------
    # Box Plot
    # -----------------------------------------

    elif chart_type == "Box Plot":

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numeric_columns:

            column = st.selectbox(
                "Select numeric column",
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

        else:

            st.warning(
                "No numeric columns available."
            )

    # -----------------------------------------
    # Pie Chart
    # -----------------------------------------

    elif chart_type == "Pie Chart":

        columns = df.columns.tolist()

        names_column = st.selectbox(
            "Select category column",
            columns
        )

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numeric_columns:

            values_column = st.selectbox(
                "Select values column",
                numeric_columns
            )

            fig = px.pie(
                df,
                names=names_column,
                values=values_column,
                title="Pie Chart"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "A numeric column is required for values."
            )

    # -----------------------------------------
    # Correlation Heatmap
    # -----------------------------------------

    elif chart_type == "Correlation Heatmap":

        numeric_df = df.select_dtypes(
            include=np.number
        )

        if numeric_df.shape[1] >= 2:

            correlation = numeric_df.corr()

            fig = px.imshow(
                correlation,
                text_auto=True,
                title="Correlation Heatmap"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "At least two numeric columns are required."
            )


# =========================================================
# AI CLEANING SUGGESTIONS
# =========================================================

elif menu == "🤖 AI Cleaning Suggestions":

    st.markdown(
        "<div class='section-title'>🤖 AI Cleaning Suggestions</div>",
        unsafe_allow_html=True
    )

    st.write(
        "CleanData AI analyzes your dataset and recommends "
        "possible data-cleaning actions."
    )

    missing = int(
        df.isnull().sum().sum()
    )

    duplicates = int(
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

    suggestions = []

    if missing > 0:

        suggestions.append(
            f"🔴 Handle {missing} missing values."
        )

    else:

        suggestions.append(
            "🟢 No missing values detected."
        )

    if duplicates > 0:

        suggestions.append(
            f"🔴 Remove {duplicates} duplicate rows."
        )

    else:

        suggestions.append(
            "🟢 No duplicate rows detected."
        )

    if outliers > 0:

        suggestions.append(
            f"🟠 Investigate approximately {outliers} outlier values."
        )

    else:

        suggestions.append(
            "🟢 No significant numeric outliers detected."
        )

    if empty_columns:

        suggestions.append(
            f"🔴 Remove empty columns: {empty_columns}"
        )

    if constant_columns:

        suggestions.append(
            f"🟠 Review constant columns: {constant_columns}"
        )

    text_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    if len(text_columns) > 0:

        suggestions.append(
            f"🔵 Consider cleaning {len(text_columns)} text columns."
        )

    for suggestion in suggestions:

        st.write(suggestion)

    st.markdown("---")

    score = calculate_quality_score(df)

    st.metric(
        "Current Data Quality Score",
        f"{score}/100"
    )


# =========================================================
# AUTOMATED CLEANING PIPELINE
# =========================================================

elif menu == "⚡ Automated Cleaning Pipeline":

    st.markdown(
        "<div class='section-title'>"
        "⚡ Automated Cleaning Pipeline"
        "</div>",
        unsafe_allow_html=True
    )

    st.write(
        "Automatically clean your dataset using "
        "multiple preprocessing steps."
    )

    remove_duplicates = st.checkbox(
        "Remove duplicate rows",
        value=True
    )

    handle_missing = st.checkbox(
        "Handle missing values automatically",
        value=True
    )

    clean_text = st.checkbox(
        "Clean text columns",
        value=True
    )

    remove_empty_columns = st.checkbox(
        "Remove completely empty columns",
        value=True
    )

    remove_outliers = st.checkbox(
        "Remove numeric outlier rows",
        value=False
    )

    if st.button(
        "⚡ Run Automated Cleaning",
        type="primary"
    ):

        before_score = calculate_quality_score(
            df
        )

        cleaned_df, report = automatic_cleaning(
            df,
            remove_duplicates,
            handle_missing,
            clean_text,
            remove_empty_columns,
            remove_outliers
        )

        after_score = calculate_quality_score(
            cleaned_df
        )

        st.session_state.df = cleaned_df

        st.session_state.auto_report = {
            "Before Quality Score": before_score,
            "After Quality Score": after_score,
            **report
        }

        save_history(
            "Automated Cleaning Pipeline",
            cleaned_df
        )

        st.success(
            "Automated cleaning completed successfully!"
        )

        st.rerun()

    if st.session_state.auto_report:

        report = st.session_state.auto_report

        st.markdown("---")

        st.subheader(
            "📊 Cleaning Results"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Before Rows",
                report["Before Rows"]
            )

        with c2:

            st.metric(
                "After Rows",
                report["After Rows"]
            )

        with c3:

            st.metric(
                "Before Missing",
                report["Before Missing Values"]
            )

        with c4:

            st.metric(
                "After Missing",
                report["After Missing Values"]
            )

        st.write(
            f"**Before Quality Score:** "
            f"{report['Before Quality Score']}/100"
        )

        st.write(
            f"**After Quality Score:** "
            f"{report['After Quality Score']}/100"
        )

        st.subheader(
            "Actions Performed"
        )

        if report["Actions"]:

            for action in report["Actions"]:

                st.write(
                    f"✅ {action}"
                )

        else:

            st.write(
                "No cleaning actions were required."
            )


# =========================================================
# DATA QUALITY DASHBOARD
# =========================================================

elif menu == "📊 Data Quality Dashboard":

    st.markdown(
        "<div class='section-title'>"
        "📊 Data Quality Dashboard"
        "</div>",
        unsafe_allow_html=True
    )

    report = generate_quality_report(
        df
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Rows",
            report["Rows"]
        )

    with c2:

        st.metric(
            "Columns",
            report["Columns"]
        )

    with c3:

        st.metric(
            "Missing Values",
            report["Missing Values"]
        )

    with c4:

        st.metric(
            "Duplicate Rows",
            report["Duplicate Rows"]
        )

    st.write("")

    c5, c6, c7, c8 = st.columns(4)

    with c5:

        st.metric(
            "Outliers",
            report["Outliers"]
        )

    with c6:

        st.metric(
            "Empty Columns",
            report["Empty Columns"]
        )

    with c7:

        st.metric(
            "Constant Columns",
            report["Constant Columns"]
        )

    with c8:

        st.metric(
            "Quality Score",
            f"{report['Quality Score']}/100"
        )

    st.markdown("---")

    st.subheader(
        "📋 Column Quality"
    )

    st.dataframe(
        detect_column_quality(df),
        use_container_width=True
    )

    st.subheader(
        "📊 Missing Values by Column"
    )

    missing_data = (
        df.isnull()
        .sum()
        .reset_index()
    )

    missing_data.columns = [
        "Column",
        "Missing Values"
    ]

    fig = px.bar(
        missing_data,
        x="Column",
        y="Missing Values",
        title="Missing Values by Column"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# EXPORT & CLEANING REPORT
# =========================================================

elif menu == "📥 Export & Cleaning Report":

    st.markdown(
        "<div class='section-title'>"
        "📥 Export & Cleaning Report"
        "</div>",
        unsafe_allow_html=True
    )

    report = generate_quality_report(
        df
    )

    st.subheader(
        "📊 Current Dataset Quality"
    )

    report_df = pd.DataFrame(
        list(report.items()),
        columns=["Metric", "Value"]
    )

    st.dataframe(
        report_df,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader(
        "📥 Download Cleaned Dataset"
    )

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download CSV",
        data=csv_data,
        file_name="cleaned_dataset.csv",
        mime="text/csv",
        use_container_width=True
    )

    try:

        excel_data = dataframe_to_excel(
            df
        )

        st.download_button(
            label="⬇️ Download Excel",
            data=excel_data,
            file_name="cleaned_dataset.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True
        )

    except Exception as e:

        st.warning(
            f"Excel export unavailable: {e}"
        )

    st.markdown("---")

    st.subheader(
        "📝 Cleaning Report"
    )

    report_text = f"""
CleanData AI - Data Cleaning Report
===================================

Rows: {report["Rows"]}
Columns: {report["Columns"]}
Missing Values: {report["Missing Values"]}
Duplicate Rows: {report["Duplicate Rows"]}
Outliers: {report["Outliers"]}
Empty Columns: {report["Empty Columns"]}
Constant Columns: {report["Constant Columns"]}
Quality Score: {report["Quality Score"]}/100

Cleaning History
================
"""

    for action in st.session_state.history:

        report_text += (
            f"- {action}\n"
        )

    st.download_button(
        label="⬇️ Download Cleaning Report",
        data=report_text,
        file_name="cleaning_report.txt",
        mime="text/plain",
        use_container_width=True
    )


# =========================================================
# HISTORY & UNDO
# =========================================================

elif menu == "🔄 History & Undo":

    st.markdown(
        "<div class='section-title'>"
        "🔄 History & Undo"
        "</div>",
        unsafe_allow_html=True
    )

    st.subheader(
        "Cleaning History"
    )

    if st.session_state.history:

        for i, action in enumerate(
            st.session_state.history,
            start=1
        ):

            st.write(
                f"**{i}.** {action}"
            )

    else:

        st.info(
            "No cleaning actions performed yet."
        )

    st.markdown("---")

    if len(
        st.session_state.dataset_versions
    ) > 1:

        if st.button(
            "↩️ Undo Last Action",
            type="primary"
        ):

            if undo_last_action():

                st.success(
                    "Last action undone successfully!"
                )

                st.rerun()

    else:

        st.info(
            "There is no action available to undo."
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#777;padding:10px;">
    🧹 <b>CleanData AI</b> |
    Intelligent Data Cleaning & Visualization Platform
    <br>
    Built with Python, Pandas, Plotly & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
