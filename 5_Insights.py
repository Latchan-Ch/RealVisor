import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

st.set_page_config(page_title="Insights & Charts", page_icon="📌")
st.title("📌 Insights & Charts")

# --------------------------- DATA LOADING ---------------------------
df = pd.read_csv('datasets/data_viz1.csv')
with open('df.pkl','rb') as file:
    df1 = pickle.load(file)

# --------------------------- STYLING ---------------------------
st.markdown("""
    <style>
    .main-title {
        font-size: 38px;
        font-weight: bold;
        color: #1F4E79;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 40px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏘️ Market Insights Dashboard</div>', unsafe_allow_html=True)

# --------------------------- VALUE-FOR-MONEY SECTORS ---------------------------
st.subheader("💸 Top Value-for-Money Sectors")
df['value_index'] = df['built_up_area'] / df['price']
value_df = df.groupby('sector')['value_index'].mean().sort_values(ascending=False).head(10).reset_index()
fig_val = px.bar(value_df, x='sector', y='value_index', color='value_index', color_continuous_scale='Viridis',
                 labels={'value_index': 'Area per Rupee'}, title='Top 10 Sectors by Area per Rupee')
st.plotly_chart(fig_val, use_container_width=True)

# --------------------------- LUXURY VS AFFORDABLE ---------------------------
st.subheader("🏷️ Luxury vs Affordable Properties")
lux_count = df1['luxury_category'].value_counts().reset_index()
lux_count.columns = ['Luxury Category', 'Count']
fig_lux = px.pie(lux_count, names='Luxury Category', values='Count', title='Property Luxury Segments')
st.plotly_chart(fig_lux, use_container_width=True)

# --------------------------- COMMON BHK AND FURNISHING ---------------------------
st.subheader("🛏️ Most Common BHKs and Furnishing Types")
col1, col2 = st.columns(2)
with col1:
    bhk_count = df['bedRoom'].value_counts().sort_index().reset_index()
    bhk_count.columns = ['BHK', 'Count']
    fig_bhk = px.bar(bhk_count, x='BHK', y='Count', title='BHK Distribution')
    st.plotly_chart(fig_bhk, use_container_width=True)

with col2:
    furnish_count = df['furnishing_type'].value_counts().reset_index()
    furnish_count.columns = ['Furnishing Type', 'Count']
    fig_furnish = px.pie(furnish_count, names='Furnishing Type', values='Count', title='Furnishing Type Share')
    st.plotly_chart(fig_furnish, use_container_width=True)

# --------------------------- TOP EXPENSIVE PROPERTIES ---------------------------
st.subheader("💰 Most Expensive Properties")
top_expensive = df.sort_values('price', ascending=False).head(10)[['sector', 'property_type', 'price', 'built_up_area']]
st.dataframe(top_expensive.style.background_gradient(cmap='Oranges'), use_container_width=True)

# --------------------------- MOST ACTIVE SECTORS ---------------------------
st.subheader("📈 Most Active Sectors (Listings Count)")
active_df = df['sector'].value_counts().head(10).reset_index()
active_df.columns = ['Sector', 'Count']
fig_active = px.bar(active_df, x='Sector', y='Count', color='Count', title='Top 10 Most Active Sectors')
st.plotly_chart(fig_active, use_container_width=True)

# --------------------------- HIGHLIGHTS ---------------------------
st.subheader("📌 Key Takeaways")
with st.expander("See Insights"):
    st.markdown("""
    - **Sector {}** offers the best value for money.
    - Most properties fall under the **{}** category.
    - **{} BHK** is the most commonly listed configuration.
    - **Sector {}** is the most active in the market.
    - High-end properties are concentrated in sectors like **{}**.
    """.format(
        value_df.iloc[0]['sector'],
        lux_count.iloc[0]['Luxury Category'],
        bhk_count.iloc[0]['BHK'],
        active_df.iloc[0]['Sector'],
        top_expensive['sector'].value_counts().idxmax()
    ))
