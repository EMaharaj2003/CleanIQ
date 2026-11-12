import io
import re

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="CleanIQ | Data Cleaning Assistant",
    page_icon="🧹",
    layout="wide",
)

st.markdown("""
<style>
    :root { color-scheme: light; }
    .stApp { background: #F5F7FB; color: #172B4D; }
    [data-testid="stHeader"] { background: rgba(245,247,251,.92); }
    h1, h2, h3, h4, p, label, .stMarkdown { color: #172B4D; }
    .welcome-text { color: #475569; font-size: 16px; }
    [data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #DCE3ED; }
    [data-testid="stSidebar"] * { color: #172B4D; }
    div[data-testid="stMetric"] {
        background: #FFFFFF; padding: 20px; border-radius: 12px;
        border: 1px solid #DCE3ED; box-shadow: 0 2px 8px rgba(23,43,77,.04);
    }
    div[data-testid="stMetric"] label, div[data-testid="stMetricLabel"] { color: #475569 !important; }
    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * { color: #172B4D !important; }
    .stButton button { border-radius: 8px; }
    [data-testid="stFileUploader"] section { background: #FFFFFF; border-color: #94A3B8; }
    [data-testid="stAlert"], [data-testid="stAlert"] > div,
    [data-testid="stAlert"] [data-testid^="stAlertContent"] {
        background: #E8F1FB !important; border-color: #B8CCE6 !important;
    }
    [data-testid="stAlert"] * { color: #172B4D !important; }
    div[data-testid="stMetricDelta"], div[data-testid="stMetricDelta"] * {
        color: #172B4D !important; background: transparent !important;
    }
    .cleaniq-table-wrap { width: 100%; overflow-x: auto; border: 1px solid #DCE3ED; border-radius: 10px; }
    .cleaniq-table { min-width: 100%; width: max-content; border-collapse: collapse; background: #FFFFFF; color: #172B4D; font-size: 14px; }
    .cleaniq-table th { background: #EAF0F8; color: #172B4D; text-align: left; }
    .cleaniq-table th, .cleaniq-table td { padding: 9px 12px; border-bottom: 1px solid #DCE3ED; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🧹 CleanIQ")
st.sidebar.caption("Agentic AI Data Cleaning Assistant")
st.sidebar.divider()
page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Dataset Upload", "Data Profiling", "AI Cleaning Agent", "Review & Approve", "Cleaning Results"],
)
st.sidebar.divider()
st.sidebar.caption("Version 0.2 | Prototype")

for key, default in {
    "dataset": None,
    "filename": None,
    "source_token": None,
    "decisions": {},
    "cleaned_dataset": None,
    "change_log": [],
    "applied_decisions": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def load_dataset(uploaded_file):
    suffix = uploaded_file.name.rsplit(".", 1)[-1].lower()
    if suffix == "csv":
        return pd.read_csv(uploaded_file)
    if suffix == "xlsx":
        return pd.read_excel(uploaded_file)
    return None


def show_dataframe(frame, *, hide_index=False, use_container_width=True, height=None):
    """Render a readable, light table consistently across browser themes."""
    html = frame.to_html(index=not hide_index, escape=True, border=0, classes="cleaniq-table")
    st.markdown(f'<div class="cleaniq-table-wrap">{html}</div>', unsafe_allow_html=True)


def assess_data_quality(df):
    rows, columns = df.shape
    cells = rows * columns
    missing = int(df.isna().sum().sum())
    completeness = 100.0 if cells == 0 else 100 * (cells - missing) / cells
    duplicate_rows = int(df.duplicated().sum())
    uniqueness = 100.0 if rows == 0 else 100 * (rows - duplicate_rows) / rows
    issues, valid_checks, invalid_values, consistency_checks, inconsistent_values = [], 0, 0, 0, 0

    email_pattern = r"^[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?(?:\.[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?)+$"
    for col in df.columns:
        values = df[col].dropna()
        name = str(col).strip().lower()
        if values.empty:
            continue
        if "email" in name or name in {"e-mail", "mail"}:
            valid_checks += len(values)
            count = int((~values.astype(str).str.strip().str.match(email_pattern, case=False, na=False)).sum())
            invalid_values += count
            if count:
                issues.append({"Issue": "Invalid email", "Column": col, "Affected values": count})
        if "date" in name or "time" in name:
            parsed = pd.to_datetime(values, errors="coerce", format="mixed")
            count = int(parsed.isna().sum())
            valid_checks += len(values)
            invalid_values += count
            if count:
                issues.append({"Issue": "Unparseable date/time", "Column": col, "Affected values": count})
        if "rating" in name or "stars" in name:
            numeric = pd.to_numeric(values, errors="coerce")
            count = int((numeric.isna() | ~numeric.between(0, 5)).sum())
            valid_checks += len(values)
            invalid_values += count
            if count:
                issues.append({"Issue": "Rating outside 0–5 or nonnumeric", "Column": col, "Affected values": count})
        if any(token in name for token in ("quantity", "qty", "amount", "price", "cost", "revenue", "total")):
            numeric = pd.to_numeric(values.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
            count = int((numeric.isna() | (numeric < 0)).sum())
            valid_checks += len(values)
            invalid_values += count
            if count:
                issues.append({"Issue": "Negative or nonnumeric quantity/price", "Column": col, "Affected values": count})
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            text = values.astype(str)
            stripped = text.str.strip()
            case_variants = stripped.str.lower().nunique() < stripped.nunique() and stripped.nunique() > 1
            inconsistent = text.ne(stripped)
            if case_variants:
                inconsistent |= stripped.ne(stripped.str.lower())
            count = int(inconsistent.sum())
            consistency_checks += len(values)
            inconsistent_values += count
            if count:
                issues.append({"Issue": "Whitespace or inconsistent text casing", "Column": col, "Affected values": count})

    validity = 100 if valid_checks == 0 else 100 * (valid_checks - invalid_values) / valid_checks
    consistency = 100 if consistency_checks == 0 else 100 * (consistency_checks - inconsistent_values) / consistency_checks
    dimensions = {
        "Completeness": round(completeness, 1),
        "Uniqueness": round(uniqueness, 1),
        "Validity": round(validity, 1),
        "Consistency": round(consistency, 1),
    }
    return {
        "score": round(sum(dimensions.values()) / len(dimensions), 1),
        "dimensions": dimensions,
        "missing": missing,
        "duplicates": duplicate_rows,
        "issues": issues,
        "validity_checks": valid_checks,
        "invalid_values": invalid_values,
        "consistency_checks": consistency_checks,
        "inconsistent_values": inconsistent_values,
    }


def build_recommendations(df):
    """Create deterministic, explainable suggestions; no changes happen here."""
    recommendations = []

    def add(category, column, count, action, text, confidence, manual=False):
        recommendations.append({
            "id": f"{category}:{column if column is not None else 'dataset'}",
            "category": category,
            "column": column,
            "count": int(count),
            "action": action,
            "recommendation": text,
            "confidence": confidence,
            "manual": manual,
        })

    for col in df.columns:
        missing = int(df[col].isna().sum())
        if missing:
            name = str(col).lower()
            if "email" in name or name in {"e-mail", "mail"}:
                add("Missing values", col, missing, None, "Review missing email addresses; copying another person's address could misattribute records.", "High", True)
            elif "date" in name or "time" in name:
                add("Missing values", col, missing, None, "Review missing dates; no replacement can be inferred safely.", "Low", True)
            elif re.search(r"(^|[_\s])id($|[_\s])", name) or name.endswith("_id"):
                add("Missing values", col, missing, None, "Review missing identifiers; generated or copied IDs could misidentify records.", "High", True)
            elif pd.api.types.is_numeric_dtype(df[col]):
                add("Missing values", col, missing, "fill_missing", "Fill blank cells with the column median.", "Medium")
            else:
                mode = df[col].dropna().mode()
                if len(mode):
                    add("Missing values", col, missing, "fill_missing", f"Fill blank cells with the most common value ({str(mode.iloc[0])[:50]}).", "Medium")
                else:
                    add("Missing values", col, missing, None, "Review blank values; there is no non-empty value to use as a suggestion.", "Low", True)

        series = df[col]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            vals = series.dropna().astype(str)
            whitespace_count = int(vals.ne(vals.str.strip()).sum())
            if whitespace_count:
                add("Formatting", col, whitespace_count, "trim_whitespace", "Remove leading and trailing spaces.", "High")

        name = str(col).strip().lower()
        nonempty = series.dropna()
        if nonempty.empty:
            continue
        if "email" in name or name in {"e-mail", "mail"}:
            pattern = r"^[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?(?:\.[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?)+$"
            bad = ~nonempty.astype(str).str.strip().str.match(pattern, case=False, na=False)
            count = int(bad.sum())
            if count:
                add("Invalid email", col, count, None, "Review invalid addresses manually; CleanIQ will not guess corrections.", "High", True)
        if "date" in name or "time" in name:
            parsed = pd.to_datetime(nonempty, errors="coerce", format="mixed")
            count = int(parsed.isna().sum())
            if count:
                add("Date validity", col, count, None, "Review unparseable dates manually; source text will be preserved.", "High", True)
        if "rating" in name or "stars" in name:
            numeric = pd.to_numeric(nonempty, errors="coerce")
            count = int((numeric.isna() | ~numeric.between(0, 5)).sum())
            if count:
                add("Rating validity", col, count, None, "Review nonnumeric or out-of-range ratings (expected 0–5).", "High", True)
        if any(token in name for token in ("quantity", "qty", "amount", "price", "cost", "revenue", "total")):
            numeric = pd.to_numeric(nonempty.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
            count = int((numeric.isna() | (numeric < 0)).sum())
            if count:
                add("Numeric validity", col, count, None, "Review negative or nonnumeric quantities/prices; values will not be changed automatically.", "High", True)

    duplicates = int(df.duplicated().sum())
    if duplicates:
        add("Duplicate rows", None, duplicates, "remove_duplicates", "Remove exact duplicate rows, keeping the first occurrence.", "High")
    return recommendations


def apply_recommendations(df, recommendations, decisions):
    cleaned = df.copy(deep=True)
    change_log = []
    approved = [r for r in recommendations if decisions.get(r["id"]) == "Approve" and r["action"]]
    # Drop exact duplicates first, then fill and format surviving rows.
    for rec in approved:
        if rec["action"] == "remove_duplicates":
            before = len(cleaned)
            cleaned = cleaned.drop_duplicates(keep="first").reset_index(drop=True)
            change_log.append({"Change": rec["recommendation"], "Column": "All columns", "Rows affected": before - len(cleaned)})
    for rec in approved:
        col, action = rec["column"], rec["action"]
        if action == "fill_missing":
            if pd.api.types.is_numeric_dtype(cleaned[col]):
                value = cleaned[col].median()
            else:
                mode = cleaned[col].dropna().mode()
                if not len(mode):
                    continue
                value = mode.iloc[0]
            count = int(cleaned[col].isna().sum())
            cleaned[col] = cleaned[col].fillna(value)
            change_log.append({"Change": f"Filled blanks with {value}", "Column": col, "Rows affected": count})
        elif action == "trim_whitespace":
            values = cleaned[col]
            mask = values.notna() & values.astype(str).ne(values.astype(str).str.strip())
            count = int(mask.sum())
            cleaned[col] = values.map(lambda value: value.strip() if isinstance(value, str) else value)
            change_log.append({"Change": "Trimmed leading/trailing spaces", "Column": col, "Rows affected": count})
    return cleaned, change_log


def current_recommendations():
    if st.session_state.dataset is None:
        return []
    return build_recommendations(st.session_state.dataset)


if page == "Overview":
    st.title("Dashboard Overview")
    st.markdown('<p class="welcome-text">Monitor dataset quality and manage your AI-powered cleaning workflow.</p>', unsafe_allow_html=True)
    st.divider()
    if st.session_state.dataset is not None:
        df = st.session_state.dataset
        assessment = assess_data_quality(df)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Data Quality Score", f"{assessment['score']}%")
        col2.metric("Total Records", f"{len(df):,}")
        col3.metric("Missing Values", f"{assessment['missing']:,}")
        col4.metric("Duplicate Rows", f"{assessment['duplicates']:,}")
        st.caption("Score is the unweighted average of completeness, uniqueness, validity, and consistency. These rule-based checks do not establish factual accuracy.")
        st.progress(assessment["score"] / 100)
        st.subheader("Dataset Summary")
        st.write(f"**File:** {st.session_state.filename}")
        st.write(f"**Columns:** {len(df.columns)}")
        st.subheader("Dataset Preview")
        show_dataframe(df.head(10))
    else:
        st.info("No dataset uploaded yet. Start by uploading a CSV or Excel file.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Data Quality Score", "--")
        col2.metric("Records Analyzed", "0")
        col3.metric("Issues Detected", "0")
        st.subheader("Getting Started")
        st.write("Upload a dataset to begin analyzing missing values, duplicates, formatting inconsistencies, and anomalies.")

elif page == "Dataset Upload":
    st.title("Upload Your Dataset")
    st.write("Upload a CSV or Excel file to begin the data quality analysis.")
    uploaded_file = st.file_uploader("Choose your dataset", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            token = (uploaded_file.name, uploaded_file.size)
            if token != st.session_state.source_token:
                df = load_dataset(uploaded_file)
                if df is None:
                    st.error("Choose a CSV or Excel file.")
                else:
                    st.session_state.dataset = df
                    st.session_state.filename = uploaded_file.name
                    st.session_state.source_token = token
                    st.session_state.decisions = {}
                    st.session_state.cleaned_dataset = None
                    st.session_state.change_log = []
                    st.session_state.applied_decisions = {}
            df = st.session_state.dataset
            if df is not None:
                st.success("Dataset uploaded successfully!")
                st.subheader("Dataset Preview")
                col1, col2, col3 = st.columns(3)
                col1.metric("Records", f"{len(df):,}")
                col2.metric("Columns", len(df.columns))
                col3.metric("Missing Values", f"{df.isna().sum().sum():,}")
                show_dataframe(df.head(20))
        except Exception as exc:
            st.error(f"Unable to read dataset: {exc}")

elif page == "Data Profiling":
    st.title("Data Profiling")
    st.write("Examine your dataset and identify potential quality issues.")
    if st.session_state.dataset is None:
        st.warning("Please upload a dataset first.")
    else:
        df = st.session_state.dataset
        profile = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isna().sum().values,
            "Unique Values": df.nunique().values,
        })
        st.subheader("Column Analysis")
        show_dataframe(profile, hide_index=True)
        assessment = assess_data_quality(df)
        st.subheader("Quality Score Breakdown")
        st.caption("Unweighted average across four dimensions. Rule-based checks do not establish factual accuracy.")
        metric_cols = st.columns(4)
        for col, (dimension, value) in zip(metric_cols, assessment["dimensions"].items()):
            col.metric(dimension, f"{value:.1f}%")
        st.progress(assessment["score"] / 100)
        st.subheader("Detected Issues")
        if assessment["issues"]:
            show_dataframe(pd.DataFrame(assessment["issues"]), hide_index=True)
        else:
            st.success("No issues found by the available checks.")
        missing = df.isna().sum()
        missing = missing[missing > 0]
        if len(missing):
            st.write("**Missing Values by Column**")
            show_dataframe(missing.rename("Missing Values").rename_axis("Column").reset_index(), hide_index=True)
        if assessment["duplicates"]:
            st.warning(f"{assessment['duplicates']} exact duplicate rows detected.")
        st.subheader("Dataset Preview")
        show_dataframe(df.head(20))

elif page == "AI Cleaning Agent":
    st.title("CleanIQ Cleaning Agent")
    st.write("Rule-based analysis creates transparent recommendations from the uploaded data. Review every proposed change before applying it.")
    if st.session_state.dataset is None:
        st.warning("Upload a dataset to activate the cleaning recommendations.")
    else:
        df = st.session_state.dataset
        recommendations = current_recommendations()
        st.success(f"Analysis complete for {st.session_state.filename}: {len(recommendations)} recommendation(s).")
        if recommendations:
            rows = [{
                "Category": r["category"], "Column": r["column"] or "All columns",
                "Affected rows/values": r["count"], "Confidence": r["confidence"],
                "Recommended action": r["recommendation"],
                "Automatic change": "No — manual review" if r["manual"] else "Available after approval",
            } for r in recommendations]
            show_dataframe(pd.DataFrame(rows), hide_index=True)
            st.info("Invalid emails, unparseable dates, out-of-range ratings, and suspicious quantities/prices are flagged for manual review. CleanIQ does not guess corrections.")
        else:
            st.success("No actionable issues were found by the current rules.")
        st.caption("This prototype uses deterministic checks, not a connected generative AI service. Column-name inference can miss fields with unfamiliar names.")

elif page == "Review & Approve":
    st.title("Review & Approve")
    st.write("Approve or reject each suggested change. Only approved automatic actions are applied.")
    if st.session_state.dataset is None:
        st.warning("Please upload a dataset first.")
    else:
        recommendations = current_recommendations()
        if not recommendations:
            st.info("There are no recommendations to review. Check Cleaning Results after analysis.")
        else:
            for i, rec in enumerate(recommendations, start=1):
                with st.container(border=True):
                    st.markdown(f"**{i}. {rec['category']}** · {rec['column'] or 'All columns'}")
                    st.write(f"{rec['recommendation']} Affected rows/values: **{rec['count']}** · Confidence: **{rec['confidence']}**")
                    if rec["manual"]:
                        st.caption("Flag only: no automatic transformation is available for this issue.")
                        st.session_state.decisions[rec["id"]] = "Reject"
                        st.selectbox("Decision", ["Manual review only"], key=f"decision_{rec['id']}", disabled=True)
                    else:
                        current = st.session_state.decisions.get(rec["id"], "Pending")
                        choice = st.selectbox("Decision", ["Pending", "Approve", "Reject"], index=["Pending", "Approve", "Reject"].index(current), key=f"decision_{rec['id']}")
                        st.session_state.decisions[rec["id"]] = choice
            st.divider()
            approved_count = sum(st.session_state.decisions.get(r["id"]) == "Approve" for r in recommendations)
            st.caption(f"{approved_count} automatic change(s) approved.")
            if st.button("Apply approved changes", type="primary", disabled=approved_count == 0):
                cleaned, log = apply_recommendations(st.session_state.dataset, recommendations, st.session_state.decisions)
                st.session_state.cleaned_dataset = cleaned
                st.session_state.change_log = log
                st.session_state.applied_decisions = dict(st.session_state.decisions)
                st.success("Approved changes applied. Review the before-and-after results.")
                st.rerun()
            if st.session_state.cleaned_dataset is not None and st.session_state.applied_decisions != st.session_state.decisions:
                st.session_state.cleaned_dataset = None
                st.session_state.change_log = []
                st.session_state.applied_decisions = {}

elif page == "Cleaning Results":
    st.title("Cleaning Results")
    st.write("Review the original and cleaned data, then download the approved result.")
    if st.session_state.dataset is None:
        st.warning("Please upload a dataset first.")
    elif st.session_state.cleaned_dataset is None:
        st.info("No approved changes have been applied yet. Review suggestions on the Review & Approve page.")
    else:
        original = st.session_state.dataset
        cleaned = st.session_state.cleaned_dataset
        before, after = st.columns(2)
        before.metric("Original rows", f"{len(original):,}")
        after.metric("Cleaned rows", f"{len(cleaned):,}", delta=f"{len(cleaned) - len(original):,}")
        before_score = assess_data_quality(original)["score"]
        after_score = assess_data_quality(cleaned)["score"]
        c1, c2 = st.columns(2)
        c1.metric("Original quality score", f"{before_score:.1f}%")
        c2.metric("Cleaned quality score", f"{after_score:.1f}%", delta=f"{after_score - before_score:+.1f}%")
        st.subheader("Approved changes")
        if st.session_state.change_log:
            show_dataframe(pd.DataFrame(st.session_state.change_log), hide_index=True)
        else:
            st.info("No automatic changes were approved; the dataset is unchanged.")
        left, right = st.columns(2)
        with left:
            st.markdown("**Original preview**")
            show_dataframe(original.head(20))
        with right:
            st.markdown("**Cleaned preview**")
            show_dataframe(cleaned.head(20))
        csv_bytes = cleaned.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Download cleaned CSV", data=csv_bytes, file_name="cleaniq_cleaned.csv", mime="text/csv")
        try:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                cleaned.to_excel(writer, index=False, sheet_name="Cleaned Data")
            st.download_button("Download cleaned Excel", data=excel_buffer.getvalue(), file_name="cleaniq_cleaned.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as exc:
            st.caption(f"Excel download is unavailable in this environment ({type(exc).__name__}); use the CSV download.")
