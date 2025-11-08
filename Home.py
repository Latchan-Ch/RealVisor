# Home.py
import streamlit as st
from PIL import Image

# Page Config
st.set_page_config(
    page_title="RealVisor | Home",
    page_icon="🔑",
    layout="wide"
)

# Optional: globally style headings to sky blue
st.markdown(
    """
    <style>
    h1, h2, h3, h4, h5, h6 { color: #4FC3F7 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Branding
st.markdown("<h1 style='text-align: center; color:#4FC3F7;'>Welcome to RealVisor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color:#555;'>Your AI-powered platform for Price Prediction, Smart Recommendations, Comparative Analytics, & Market Insights</h4>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar navigation info
st.sidebar.image("https://img.freepik.com/premium-vector/real-estate-logo-design-with-white-background_121452-274.jpg", width=1000)
st.sidebar.title("🔑 RealVisor")
st.sidebar.markdown("### Navigate To:")
st.sidebar.markdown("- 💸 Price Prediction")
st.sidebar.markdown("- 📊 Market Analytics")
st.sidebar.markdown("- 🏢 Housing Recommendations")
st.sidebar.markdown("- ⚖️ Compare Properties")
st.sidebar.markdown("- 📌 Insights & Charts")

# Home page interactive buttons (5 columns)
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.image("https://img.icons8.com/win10/1200/bag-diagonal-view.jpg", width=90)
    st.subheader("💸 Price Prediction")
    st.write("Predict housing prices using advanced ML models.")
    if st.button("Go to Price Prediction", key="btn_price_prediction"):
        st.switch_page("pages/1_Price_Prediction.py")

with col2:
    st.image("https://static.vecteezy.com/system/resources/previews/043/916/424/non_2x/market-research-glyph-inverted-icon-design-vector.jpg", width=90)
    st.subheader("📊 Market Analysis")
    st.write("Visualize pricing trends and understand key patterns.")
    if st.button("Go to Market Analysis", key="btn_market_analysis"):
        st.switch_page("pages/2_Analysis_Tool.py")

with col3:
    st.image("https://img.icons8.com/deco-glyph/1200/search-property.jpg", width=90)
    st.subheader("🏢 Apartment Recommendations")
    st.write("Get personalized property suggestions based on your needs.")
    if st.button("Go to Recommendations", key="btn_recommendations"):
        st.switch_page("pages/3_Recommend_Apartments.py")

with col4:
    st.image("https://static.vecteezy.com/system/resources/previews/014/641/138/non_2x/compare-prices-line-icon-vector.jpg", width=90)
    st.subheader("⚖️ Compare Properties")
    st.write("Compare up to 3 properties side-by-side.")
    if st.button("Go to Compare Properties", key="btn_compare_properties"):
        st.switch_page("pages/4_Compare_Properties.py")

with col5:
    st.image("https://cdn-icons-png.flaticon.com/512/4341/4341139.png", width=90)
    st.subheader("📌 Insights & Charts")
    st.write("Explore location-based insights, price trends, and charts.")
    if st.button("Go to Insights & Charts", key="btn_insights_charts"):
        st.switch_page("pages/5_Insights.py")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center;'><h3 style='color:#4FC3F7;'>Let RealVisor be your companion to smart property decisions.</h3></div>", unsafe_allow_html=True)
