# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import plotly.express as px
# import pickle
# from wordcloud import WordCloud
# import seaborn as sns
#
# st.set_page_config(
#     page_title="Plotting Demo"
# )
#
# st.title('Analysis')
#
# new_df = pd.read_csv('datasets/data_viz1.csv')
# feature_text = pickle.load(open('datasets/feature_text.pkl','rb'))
#
# group_df = new_df.groupby('sector')[['price', 'price_per_sqft', 'built_up_area', 'latitude', 'longitude']].mean()
#
# st.header('Sectors Price Per Sqft Geomap')
# fig = px.scatter_mapbox(group_df, lat="latitude", lon="longitude", color="price_per_sqft", size='built_up_area',
#                   color_continuous_scale=px.colors.cyclical.IceFire, zoom=10,
#                   mapbox_style="open-street-map", width=1200, height=700, hover_name=group_df.index)
# st.plotly_chart(fig,use_container_width=True)
#
# st.header('Features Wordcloud')
#
# wordcloud = WordCloud(width = 800, height = 800,
#                       background_color ='white',
#                       stopwords = set(['s']),  # Any stopwords you'd like to exclude
#                       min_font_size = 10).generate(feature_text)
#
# fig = plt.figure(figsize = (8, 8), facecolor = None)
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.tight_layout(pad = 0)
# st.pyplot(fig)
#
# st.header('Area Vs Price')
#
# property_type = st.selectbox('Select Property Type', ['Flat', 'House'])
#
# if property_type == 'Flat':
#     fig1 = px.scatter(new_df[new_df['property_type'] == 'flat'], x="built_up_area", y="price", color="bedRoom", title="Area Vs Price")  # Show the plot
#     st.plotly_chart(fig1, use_container_width=True)
# else:
#     fig1 = px.scatter(new_df[new_df['property_type'] == 'house'], x="built_up_area", y="price", color="bedRoom", title="Area Vs Price")  # Show the plot
#     st.plotly_chart(fig1, use_container_width=True)
#
# st.header('BHK Pie Chart')
#
# sector_options = new_df['sector'].unique().tolist()
# sector_options.insert(0,'overall')
# selected_sector = st.selectbox('Select Sector', sector_options)
# if selected_sector == 'overall':
#     fig2 = px.pie(new_df, names='bedRoom')  # Show the plot
#     st.plotly_chart(fig2, use_container_width=True)
# else:
#     fig2 = px.pie(new_df[new_df['sector'] == selected_sector], names='bedRoom')  # Show the plot
#     st.plotly_chart(fig2, use_container_width=True)
#
# st.header('Side By Side BHK Price Comparison')
#
# fig3 = px.box(new_df[new_df['bedRoom'] <=4], x='bedRoom', y='price')  # Show the plot
# st.plotly_chart(fig3, use_container_width=True)
#
# st.header('Side By Side Distplot For Property Type')
#
# fig4, ax = plt.subplots(figsize=(8, 5))
# sns.histplot(new_df[new_df['property_type'] == 'house']['price'], kde=True, label='House', ax=ax)
# sns.histplot(new_df[new_df['property_type'] == 'flat']['price'], kde=True, label='Flat', ax=ax)
# ax.set_title('Price Distribution: House vs Flat')
# ax.set_xlabel('Price')
# ax.set_ylabel('Density')
# ax.legend()
# st.pyplot(fig4)


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
from wordcloud import WordCloud
import seaborn as sns

st.set_page_config(page_title="Analysis Tool", page_icon="📊")
st.title("📊 Analysis Tool")

# --------------------------- DATA LOADING ---------------------------

new_df = pd.read_csv('datasets/data_viz1.csv')
feature_text = pickle.load(open('datasets/feature_text.pkl', 'rb'))
group_df = new_df.groupby('sector')[['price', 'price_per_sqft', 'built_up_area', 'latitude', 'longitude']].mean()

# --------------------------- HEADER ---------------------------

def show_header():
    st.markdown("""
        <style>
            .main-title {
                font-size: 36px;
                font-weight: bold;
                color: #34495E;
                text-align: center;
                margin-bottom: 40px;
            }
        </style>
        <div class="main-title">📊 Property Market Insights</div>
    """, unsafe_allow_html=True)

# --------------------------- COMPONENTS ---------------------------

def geomap_plot():
    st.subheader('Sectors Price Per Sqft Geomap')
    fig = px.scatter_mapbox(
        group_df, lat="latitude", lon="longitude", color="price_per_sqft", size='built_up_area',
        color_continuous_scale=px.colors.cyclical.IceFire, zoom=10,
        mapbox_style="open-street-map", width=1200, height=700, hover_name=group_df.index
    )
    st.plotly_chart(fig, use_container_width=True)

def wordcloud_plot():
    st.subheader('Features Wordcloud')
    wordcloud = WordCloud(
        width=800, height=800, background_color='white',
        stopwords=set(['s']), min_font_size=10
    ).generate(feature_text)

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(fig)

def area_vs_price_plot():
    st.subheader('Area vs Price')
    property_type = st.selectbox('Select Property Type', ['Flat', 'House'])
    df = new_df[new_df['property_type'] == property_type.lower()]
    fig = px.scatter(df, x="built_up_area", y="price", color="bedRoom", title="Area Vs Price")
    st.plotly_chart(fig, use_container_width=True)

def bhk_pie_chart():
    st.subheader('BHK Pie Chart')
    sector_options = ['overall'] + sorted(new_df['sector'].unique().tolist())
    selected_sector = st.selectbox('Select Sector', sector_options, key='sector_pie')

    df = new_df if selected_sector == 'overall' else new_df[new_df['sector'] == selected_sector]
    fig = px.pie(df, names='bedRoom', title=f'BHK Distribution: {selected_sector.title()}')
    st.plotly_chart(fig, use_container_width=True)

def bhk_price_box():
    st.subheader('Side by Side BHK Price Comparison')
    fig = px.box(new_df[new_df['bedRoom'] <= 4], x='bedRoom', y='price')
    st.plotly_chart(fig, use_container_width=True)

def property_type_distplot():
    st.subheader('Side by Side Distplot for Property Type')
    sector_options = ['overall'] + sorted(new_df['sector'].unique().tolist())
    selected_sector = st.selectbox('Select Sector', sector_options, key='sector_displot')
    fig, ax = plt.subplots(figsize=(8, 5))
    if selected_sector == 'overall':
        sns.histplot(new_df[new_df['property_type'] == 'house']['price'],
                     kde=True, label='House', ax=ax)
        sns.histplot(new_df[new_df['property_type'] == 'flat']['price'],
                     kde=True, label='Flat', ax=ax)
    else:
        sns.histplot(new_df[(new_df['property_type'] == 'house') & (new_df['sector'] == selected_sector)]['price'],
                     kde=True, label='House', ax=ax)
        sns.histplot(new_df[(new_df['property_type'] == 'flat') & (new_df['sector'] == selected_sector)]['price'],
                     kde=True, label='Flat', ax=ax)

    ax.set_title('Price Distribution: House vs Flat')
    ax.set_xlabel('Price')
    ax.set_ylabel('Density')
    ax.legend()
    st.pyplot(fig)

# --------------------------- MAIN APP ---------------------------

show_header()
geomap_plot()
wordcloud_plot()
area_vs_price_plot()
bhk_pie_chart()
bhk_price_box()
property_type_distplot()