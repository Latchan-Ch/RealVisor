# import streamlit as st
# import pickle
# import pandas as pd
# import numpy as np
#
# st.set_page_config(
#     page_title="Viz Demo"
# )
#
# with open('df.pkl','rb') as file:
#     df = pickle.load(file)
#
# with open('pipeline.pkl','rb') as file:
#     pipeline = pickle.load(file)
#
# # property_type
# property_type = st.selectbox('Property Type',['flat', 'house'])
#
# # sector
# sector = st.selectbox('Sector', sorted(df['sector'].unique().tolist()))
#
# bedrooms = float(st.selectbox('Number of Bedrooms', sorted(df['bedRoom'].unique().tolist())))
#
# bathrooms = float(st.selectbox('Number of bathrooms', sorted(df['bathroom'].unique().tolist())))
#
# balconies = st.selectbox('Balconies', sorted(df['balcony'].unique().tolist()))
#
# property_age = st.selectbox('Property Age', sorted(df['agePossession'].unique().tolist()))
#
# built_up_area = float(st.number_input('Built up area'))
#
# servant_room = float(st.selectbox('Servant Room', [0.0,1.0]))
#
# store_room = float(st.selectbox('Store Room', [0.0,1.0]))
#
# furnishing_type = st.selectbox('Furnishing Type', sorted(df['furnishing_type'].unique().tolist()))
#
# luxury_category = st.selectbox('Luxury Category', sorted(df['luxury_category'].unique().tolist()))
#
# floor_category = st.selectbox('Floor Category', sorted(df['floor_category'].unique().tolist()))
#
# if st.button('Predict'):
#
#     # form a dataframe
#     data = [[property_type, sector, bedrooms, bathrooms, balconies, property_age, built_up_area, servant_room, store_room, furnishing_type, luxury_category, floor_category]]
#     columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
#                'agePossession', 'built_up_area', 'servant room', 'store room',
#                'furnishing_type', 'luxury_category', 'floor_category']
#
#     # Convert to DataFrame
#     one_df = pd.DataFrame(data, columns=columns)
#
#     # predict
#     base_price = np.expm1(pipeline.predict(one_df))[0]
#     low = base_price -0.22
#     high = base_price + 0.22
#
#     # display
#     st.text('The price of the {} is between {} Cr and {} Cr'.format(property_type, round(low,2), round(high,2)))


import streamlit as st
import pickle
import pandas as pd
import numpy as np
import joblib


st.set_page_config(page_title="Price Prediction", page_icon="💸")
st.title("💸 Price Prediction")


# -------------------- Utilities --------------------

def get_price_column(df):
    """
    Detect a price-like numeric column in df.
    Returns column name or None.
    """
    candidates = [
        'price', 'Price', 'amount', 'Amount', 'value', 'Value',
        'sale_price', 'salePrice', 'price_inr', 'price_rs', 'price_cr',
        'final_price', 'listing_price'
    ]
    for c in candidates:
        if c in df.columns:
            return c

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if not numeric_cols:
        return None

    exclude_keywords = [
        'area', 'built', 'sq', 'sqft', 'bed', 'bath', 'balcony', 'room', 'store',
        'servant', 'floor', 'age', 'year', 'distance', 'lat', 'lon', 'latitude', 'longitude', 'count'
    ]
    numeric_filtered = []
    for c in numeric_cols:
        lower = c.lower()
        if any(k in lower for k in exclude_keywords):
            continue
        numeric_filtered.append(c)

    candidates_to_consider = numeric_filtered if numeric_filtered else numeric_cols

    means = {}
    for c in candidates_to_consider:
        try:
            means[c] = float(df[c].abs().mean())
        except Exception:
            means[c] = 0.0

    best = max(means, key=means.get)
    if means.get(best, 0) <= 0:
        return None
    return best

# -------------------- EMI UI --------------------

def emi_calculator_ui(default_price=None):
    """
    EMI calculator with its own form (so typing doesn't cause immediate reruns).
    default_price should be in rupees (int) or None.
    """
    st.subheader("💰 EMI / Loan Estimator")

    with st.form("emi_form"):
        default_price_val = int(default_price) if default_price is not None else 1000000
        price_for_emi = st.number_input(
            "Property Price for EMI calculation (₹)",
            value=default_price_val,
            step=10000,
            format="%d"
        )
        down_payment = st.number_input("Down Payment (₹)", value=0, step=10000, format="%d")
        interest = st.number_input("Annual Interest Rate (%)", min_value=0.1, max_value=25.0, value=8.0, step=0.1)
        years = st.number_input("Loan Tenure (Years)", min_value=1, max_value=30, value=20)

        calculate = st.form_submit_button("Calculate EMI")

    if calculate:
        loan_amount = max(price_for_emi - down_payment, 0)
        monthly_interest = interest / (12 * 100)
        months = int(years * 12)
        if months == 0:
            st.error("Loan tenure must be at least 1 year.")
            return
        if monthly_interest == 0:
            emi = loan_amount / months
        else:
            emi = (loan_amount * monthly_interest * (1 + monthly_interest) ** months) / ((1 + monthly_interest) ** months - 1)
        total_payment = emi * months + down_payment
        st.success(f"Estimated EMI: ₹{emi:,.0f} / month")
        st.write(f"Total repayment (incl. down payment): ₹{total_payment:,.0f} over {years} years")

# -------------------- Top-5 Dashboard --------------------

def top5_dashboard(df, by='avg_price'):
    st.subheader("📈 Top 5 Locations")
    price_col = get_price_column(df)
    if price_col is None:
        st.warning("Cannot compute Top-5: no price-like column found in dataset.")
        return

    loc_col = 'location' if 'location' in df.columns else ('sector' if 'sector' in df.columns else df.columns[0])
    if by == 'avg_price':
        top5 = df.groupby(loc_col)[price_col].mean().sort_values(ascending=False).head(5)
        st.bar_chart(top5)
        st.write("Top 5 locations by average price (descending).")
    elif by == 'growth':
        if 'date' not in df.columns:
            st.warning("Growth option requires a 'date' column in the dataset.")
            return
        growth = {}
        for loc, sub in df.groupby(loc_col):
            sub_sorted = sub.sort_values('date')
            if len(sub_sorted) < 2:
                continue
            first = sub_sorted[price_col].iloc[0]
            last = sub_sorted[price_col].iloc[-1]
            growth[loc] = (last / first) - 1 if first > 0 else 0
        growth_series = pd.Series(growth).sort_values(ascending=False).head(5)
        st.bar_chart((growth_series * 100).round(2))
        st.write("Top 5 locations by price growth (%)")

# -------------------- UI / Forms --------------------

def show_header():
    st.markdown("""
        <style>
            .main-title {
                font-size: 36px;
                font-weight: 700;
                color: #2C3E50;
                text-align: center;
                margin-bottom: 30px;
            }
        </style>
        <div class="main-title">🏠 Property Price Estimator</div>
    """, unsafe_allow_html=True)

def input_form(df):
    with st.form("price_prediction_form"):
        property_type = st.selectbox('Property Type', ['flat', 'house'])

        col1, col2 = st.columns(2)
        sector = col1.selectbox('Sector', sorted(df['sector'].unique()))
        property_age = col2.selectbox('Property Age', sorted(df['agePossession'].unique()))

        col1, col2, col3 = st.columns(3)
        bedrooms = float(col1.selectbox('Bedrooms', sorted(df['bedRoom'].unique())))
        bathrooms = float(col2.selectbox('Bathrooms', sorted(df['bathroom'].unique())))
        balconies = col3.selectbox('Balconies', sorted(df['balcony'].unique()))

        built_up_area = st.number_input('Built-up Area (in sq. ft)', min_value=100.0, step=10.0)

        col1, col2 = st.columns(2)
        servant_room = float(col1.selectbox('Servant Room', [0.0, 1.0]))
        store_room = float(col2.selectbox('Store Room', [0.0, 1.0]))

        col1, col2 = st.columns(2)
        furnishing_type = col1.selectbox('Furnishing Type', sorted(df['furnishing_type'].unique()))
        luxury_category = col2.selectbox('Luxury Category', sorted(df['luxury_category'].unique()))

        floor_category = st.selectbox('Floor Category', sorted(df['floor_category'].unique()))

        submitted = st.form_submit_button("Predict Price ")

        input_data = {
            'property_type': property_type,
            'sector': sector,
            'bedRoom': bedrooms,
            'bathroom': bathrooms,
            'balcony': balconies,
            'agePossession': property_age,
            'built_up_area': built_up_area,
            'servant room': servant_room,
            'store room': store_room,
            'furnishing_type': furnishing_type,
            'luxury_category': luxury_category,
            'floor_category': floor_category
        }

        return submitted, input_data

# -------------------- Main logic --------------------

st.set_page_config(page_title="Price Prediction", layout="centered")

# Load data + model
with open('df.pkl', 'rb') as f:
    df = pickle.load(f)

with open('pipeline_compressed.pkl', 'rb') as f:
    pipeline = joblib.load(f)

show_header()

submitted, input_data = input_form(df)

if submitted:
    # build one-row dataframe with required columns
    one_df = pd.DataFrame([input_data])

    # predict (pipeline returns log-price in this repo)
    predicted_log_price = pipeline.predict(one_df)[0]
    base_price = np.expm1(predicted_log_price)   # in crores

    low = round(base_price - 0.22, 2)
    high = round(base_price + 0.22, 2)

    st.success(f" Estimated Price Range: **₹{low} Cr - ₹{high} Cr**")

    # store last prediction in session_state (so EMI UI persists across reruns)
    try:
        pred_price_cr = float(base_price)
        pred_price_rupees = int(round(pred_price_cr * 10_000_000))  # 1 Cr = 10,000,000 INR
    except Exception:
        pred_price_cr = None
        pred_price_rupees = None

    location = input_data.get('sector') or input_data.get('location')

    st.session_state['last_prediction'] = {
        'base_price_cr': pred_price_cr,
        'base_price_rupees': pred_price_rupees,
        'low_cr': low,
        'high_cr': high,
        'location': location
    }

    # Show EMI calculator prefilled with predicted price (if available)
    try:
        emi_calculator_ui(default_price=pred_price_rupees)
    except Exception as e:
        st.warning("EMI calculator error: " + str(e))

    # Show Top 5 locations
    try:
        top5_dashboard(df, by='avg_price')
    except Exception as e:
        st.warning("Top-5 dashboard error: " + str(e))

# Persisted UI when reopening / navigating (so user doesn't lose prediction on reruns)
if 'last_prediction' in st.session_state:
    lp = st.session_state['last_prediction']

    # show persisted estimated price
    if lp.get('low_cr') is not None and lp.get('high_cr') is not None:
        try:
            st.success(f" Estimated Price Range: **₹{lp['low_cr']} Cr - ₹{lp['high_cr']} Cr**")
        except Exception:
            st.success(f" Estimated Price Range: {lp.get('low_cr')} - {lp.get('high_cr')}")

    # show EMI form with persisted predicted price
    try:
        default_rupees = lp.get('base_price_rupees', None)
        emi_calculator_ui(default_price=default_rupees)
    except Exception:
        st.info("EMI estimator unavailable.")

    # show top-5
    try:
        top5_dashboard(df, by='avg_price')
    except Exception:
        st.info("Top-5 dashboard unavailable.")
