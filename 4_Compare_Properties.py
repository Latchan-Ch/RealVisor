# pages/5_Compare_Properties.py
import streamlit as st
import pickle
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Compare Properties", page_icon="⚖️")
st.title("⚖️ Compare Properties")


# -----------------------
# Load data & model
# -----------------------
@st.cache_data
def load_data():
    try:
        with open("df.pkl", "rb") as f:
            df = pickle.load(f)
    except Exception as e:
        st.error("Couldn't load df.pkl. Make sure file exists in repo root.")
        raise e
    return df

@st.cache_resource
def load_pipeline():
    try:
        pipe = joblib.load("pipeline_compressed.pkl")
        return pipe
    except Exception:
        return None

df = load_data()
pipeline = load_pipeline()

# helper to create a friendly label for selection
def make_label(row, idx):
    parts = []
    # add property name if exists
    if 'property_name' in row and pd.notna(row['property_name']) and str(row['property_name']).strip():
        parts.append(str(row['property_name']))
    # always include index and sector / type
    if 'sector' in row and pd.notna(row['sector']):
        parts.append(str(row['sector']))
    if 'property_type' in row and pd.notna(row['property_type']):
        parts.append(str(row['property_type']))
    # include price if present
    if 'price' in row and pd.notna(row['price']):
        parts.append(f"price={row['price']}")
    label = f"{idx} | " + " • ".join(parts)
    return label

# Build selection list
records = []
labels = []
for idx, r in df.reset_index().iterrows():
    label = make_label(r, r.get('index', idx))
    records.append(r.get('index', idx))
    labels.append(label)

# UI: multi-select (limit 3)
st.write("Search and select up to **3** properties to compare.")
selected_labels = st.multiselect("Select properties", options=labels, max_selections=3)

# map back selected labels to dataframe indices
selected_idx = []
label_to_idx = dict(zip(labels, records))
for lbl in selected_labels:
    if lbl in label_to_idx:
        selected_idx.append(label_to_idx[lbl])

if not selected_idx:
    st.info("No properties selected yet. Pick up to 3 from the box above.")
    st.stop()

# Build comparison df (take rows from original df by index)
comp_df = df.loc[df.index.isin(selected_idx)].copy()
if comp_df.empty:
    st.error("No matching rows found for selected properties.")
    st.stop()

# Choose which columns to display (only those present)
preferred_cols = [
    "property_name", "sector", "property_type", "bedRoom", "bathroom",
    "built_up_area", "price", "price_per_sqft", "furnishing_type", "luxury_category", "agePossession"
]
cols_to_show = [c for c in preferred_cols if c in comp_df.columns]

# compute price_per_sqft if not present and price & built_up_area exist
if "price_per_sqft" not in comp_df.columns and "price" in comp_df.columns and "built_up_area" in comp_df.columns:
    # guess: price might be in Crores; try to detect scale (if mean < 100 then likely Crores)
    mean_price = comp_df["price"].abs().mean()
    if pd.notna(mean_price) and mean_price > 0:
        if mean_price < 1000:
            rupee_multiplier = 10_000_000
        else:
            rupee_multiplier = 1
    else:
        rupee_multiplier = 10_000_000
    try:
        comp_df["price_per_sqft"] = (comp_df["price"].astype(float) * rupee_multiplier) / comp_df["built_up_area"].replace(0, np.nan)
        if "price_per_sqft" not in cols_to_show:
            cols_to_show.append("price_per_sqft")
    except Exception:
        pass

# Reorder columns so index is visible
display_df = comp_df[cols_to_show].copy()
display_df.index = display_df.index.astype(str)

# Show side-by-side table
st.subheader("Comparison Table")
st.dataframe(display_df.style.format(precision=2), use_container_width=True)

# Quick metrics: predicted price (if pipeline available)
st.subheader("Quick Metrics")
col1, col2 = st.columns([1,2])

with col1:
    st.write("Selected properties:")
    for i, idx in enumerate(comp_df.index):
        st.markdown(f"- **{i+1}.** index: `{idx}`")
with col2:
    # Try model prediction if pipeline exists
    if pipeline is not None:
        try:
            # Attempt to predict on comp_df; pipeline may expect specific columns — best effort:
            preds_log = pipeline.predict(comp_df)
            preds = np.expm1(preds_log)  # predicted price in the same unit your pipeline was trained to output (we expect Crores)
            # attach to comp_df for downstream use
            comp_df["_pred_price_model"] = preds
            # nicely formatted display: crores + rupee
            pr = comp_df[["_pred_price_model"]].copy()
            pr.index = pr.index.astype(str)
            pr_display = pr.copy()
            pr_display["_pred_price_model"] = pr_display["_pred_price_model"].apply(
                lambda x: f"₹{x:.2f} Cr ({int(round(x * 10_000_000)):,} INR)"
            )
            st.write("Model predicted price (approx):")
            st.dataframe(pr_display, use_container_width=True)
        except Exception:
            # If the model can't predict, don't show a big blue info box; show a small caption instead
            st.caption("Model prediction unavailable for the selected rows (pipeline input mismatch).")
    else:
        st.caption("No pipeline found (pipeline_compressed.pkl), model predictions unavailable.")

# Visual: price and psf comparison bar chart
plot_cols = []
if "price" in comp_df.columns:
    plot_cols.append("price")
if "price_per_sqft" in comp_df.columns:
    plot_cols.append("price_per_sqft")

if plot_cols:
    st.subheader("Visual comparison")
    plot_data = comp_df[plot_cols].reset_index().melt(id_vars="index", value_vars=plot_cols,
                                                     var_name="metric", value_name="value")
    fig = px.bar(plot_data, x="index", y="value", color="metric", barmode="group",
                 labels={"index":"property_index","value":"value"}, title="Price / Price-per-sqft comparison")
    st.plotly_chart(fig, use_container_width=True)
else:
    # Replace big blue info with subtle caption
    st.caption("Numeric price or price_per_sqft not available for visual comparison.")

# --------------------------
# New explainable heuristic
# --------------------------
def compute_property_score(row, df_all, preds_col="_pred_price_model"):
    """
    Returns a float score 0-10 for a single property `row` (pandas Series).
    Uses:
      - value_score (0..5) comparing local avg PSF vs predicted PSF
      - beds_score (0..2)
      - age_score (0..1.5)
      - furnish_score (0..1.5)
    """
    # Predicted price in Crores (pipeline), if present on row
    pred_price_cr = None
    if preds_col in row and pd.notna(row[preds_col]):
        try:
            pred_price_cr = float(row[preds_col])
        except Exception:
            pred_price_cr = None

    # predicted psf (rupees per sqft)
    predicted_psf = None
    if pred_price_cr is not None and 'built_up_area' in row and pd.notna(row['built_up_area']) and float(row['built_up_area']) > 0:
        pred_price_rupees = pred_price_cr * 10_000_000
        predicted_psf = pred_price_rupees / float(row['built_up_area'])

    # compute local average psf using df_all if possible (sector-based)
    area_avg_psf = None
    loc_col = 'sector' if 'sector' in df_all.columns else ( 'location' if 'location' in df_all.columns else None )
    if loc_col and loc_col in row and pd.notna(row[loc_col]):
        loc = row[loc_col]
        sub = df_all[df_all[loc_col] == loc]
    else:
        sub = df_all

    # try to compute average psf from price & built_up_area in the global df
    try:
        if 'price' in sub.columns and 'built_up_area' in sub.columns and (sub['built_up_area'] > 0).any():
            # assume price in crores like your pipeline -> convert to rupees
            avg_price = sub['price'].astype(float).mean()
            median_area = sub['built_up_area'].replace(0, np.nan).median()
            if pd.notna(avg_price) and median_area > 0:
                area_avg_psf = (avg_price * 10_000_000) / median_area
    except Exception:
        area_avg_psf = None

    # fallback: if comp_df has price_per_sqft values, use their median
    if area_avg_psf is None:
        try:
            if 'price_per_sqft' in sub.columns and (sub['price_per_sqft'].dropna().size > 0):
                area_avg_psf = float(sub['price_per_sqft'].median())
        except Exception:
            area_avg_psf = None

    # ----- build component scores -----
    # Value score: scale area_avg_psf / predicted_psf into 0..5
    value_score = 2.5  # neutral by default
    if predicted_psf and area_avg_psf and predicted_psf > 0:
        ratio = area_avg_psf / predicted_psf  # >1 => property cheaper than area average -> good
        r = min(max(ratio, 0.5), 3.0)
        value_score = ((r - 0.5) / (3.0 - 0.5)) * 5.0

    # Bedrooms score: more beds -> up to 2 points (cap at 4)
    beds = float(row.get('bedRoom') or 0)
    beds_score = min(beds / 4.0 * 2.0, 2.0)

    # Age score: newer = better, simple mapping from agePossession text
    age_text = str(row.get('agePossession') or "").lower()
    if "new" in age_text:
        age_score = 1.5
    elif "relatively" in age_text or "recent" in age_text:
        age_score = 1.0
    elif "moderate" in age_text or "moderately" in age_text:
        age_score = 0.75
    else:
        age_score = 0.25

    # Furnishing/luxury score
    furnishing = str(row.get('furnishing_type') or "").lower()
    luxury = str(row.get('luxury_category') or "").lower()
    furn_score = 0.5 if "unfurnished" in furnishing or furnishing == "" else (1.0 if "semi" in furnishing else 1.5)
    lux_score = 0.0
    if "high" in luxury or "luxury" in luxury:
        lux_score = 1.0
    elif "medium" in luxury:
        lux_score = 0.5
    furnish_score = min(furn_score + lux_score, 1.5)

    total = value_score + beds_score + age_score + furnish_score
    total = max(0.0, min(10.0, total))
    # Also return components if you want to show the breakdown later
    return round(total, 2), dict(
        value_score=round(value_score,2),
        beds_score=round(beds_score,2),
        age_score=round(age_score,2),
        furnish_score=round(furnish_score,2),
        predicted_psf=round(predicted_psf,2) if predicted_psf is not None else None,
        area_avg_psf=round(area_avg_psf,2) if area_avg_psf is not None else None
    )

# Compute scores for selected properties
st.subheader("Recommendation (explainable heuristic)")
with st.expander("How this works"):
    st.write("""
      Score components (total 0–10):
      - **Value-for-money (0–5)**: compares local avg PSF vs property predicted PSF (lower PSF → larger points).
      - **Bedrooms (0–2)**: more bedrooms → small boost.
      - **Age (0–1.5)**: newer properties score higher.
      - **Furnishing/Luxury (0–1.5)**: nicer furnishing increases score.
    """)

scores = []
scores_details = {}
# Ensure we have model predictions attached if possible (we attempted earlier)
if "_pred_price_model" not in comp_df.columns and pipeline is not None:
    try:
        preds_log = pipeline.predict(comp_df)
        comp_df["_pred_price_model"] = np.expm1(preds_log)
    except Exception:
        # if pipeline fails, continue - compute will have fewer data points for PSF
        pass

for idx, row in comp_df.iterrows():
    sc, details = compute_property_score(row, df)
    scores.append((idx, sc))
    scores_details[idx] = details

scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
if scores_sorted:
    st.write("Ranked (best → worst) by heuristic:")
    for rank, (idx, sc) in enumerate(scores_sorted, start=1):
        det = scores_details.get(idx, {})
        # show score plus a tiny breakdown
        st.markdown(f"**{rank}.** index `{idx}` — score: **{sc:.2f} / 10**")
        st.caption(f"Breakdown (value:{det.get('value_score')}, beds:{det.get('beds_score')}, age:{det.get('age_score')}, furnish:{det.get('furnish_score') if 'furnish_score' in det else det.get('furnish_score')}) • "
                   f"predicted_psf: {det.get('predicted_psf')} • local_avg_psf: {det.get('area_avg_psf')}")
else:
    st.info("No properties to score.")

