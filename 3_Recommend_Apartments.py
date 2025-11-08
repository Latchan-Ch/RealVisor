# import streamlit as st
# import pickle
# import pandas as pd
# import numpy as np
#
# st.set_page_config(
#     page_title="Recommend Apartments"
# )
#
# location_df = pickle.load(open('datasets/location_distance.pkl', 'rb'))
# cosine_sim1 = pickle.load(open('datasets/cosine_sim1.pkl', 'rb'))
# cosine_sim2 = pickle.load(open('datasets/cosine_sim2.pkl', 'rb'))
# cosine_sim3 = pickle.load(open('datasets/cosine_sim3.pkl', 'rb'))
#
#
# def recommend_properties_with_scores(property_name, top_n=5):
#     cosine_sim_matrix = 0.5 * cosine_sim1 + 0.8 * cosine_sim2 + 1 * cosine_sim3
#     # cosine_sim_matrix = cosine_sim3
#
#     # Get the similarity scores for the property using its name as the index
#     sim_scores = list(enumerate(cosine_sim_matrix[location_df.index.get_loc(property_name)]))
#
#     # Sort properties based on the similarity scores
#     sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#
#     # Get the indices and scores of the top_n most similar properties
#     top_indices = [i[0] for i in sorted_scores[1:top_n + 1]]
#     top_scores = [i[1] for i in sorted_scores[1:top_n + 1]]
#
#     # Retrieve the names of the top properties using the indices
#     top_properties = location_df.index[top_indices].tolist()
#
#     # Create a dataframe with the results
#     recommendations_df = pd.DataFrame({
#         'PropertyName': top_properties,
#         'SimilarityScore': top_scores
#     })
#
#     return recommendations_df
#
# # --- Initialize session state ---
# if 'search_clicked' not in st.session_state:
#     st.session_state.search_clicked = False
# if 'results' not in st.session_state:
#     st.session_state.results = None
# if 'selected_apartment' not in st.session_state:
#     st.session_state.selected_apartment = None
#
# st.title('Select Location And Radius')
#
# selected_location = st.selectbox('Location', sorted(location_df.columns.tolist()))
#
# radius = st.number_input('Radius in km')
#
# selected_apartment = None
#
# if st.button('Search'):
#     result_ser = location_df[location_df[selected_location] <= radius*1000][selected_location].sort_values()
#     if result_ser.empty:
#         st.warning("No apartments found within the given radius")
#         st.session_state.search_clicked = False
#     else:
#         st.session_state.search_clicked = True
#         st.session_state.results = result_ser
#         st.session_state.selected_apartment = list(result_ser.index)[0]  # Select default
#
#
# if st.session_state.search_clicked and st.session_state.results is not None:
#     apartment = []
#     distance = []
#     for key, value in st.session_state.results.items():
#         apartment.append(key)
#         distance.append(str(value / 1000) + ' km')
#     st.session_state.selected_apartment = st.radio(
#         "Select anyone for recommendation",
#         apartment,
#         captions=distance
#     )
#     if st.session_state.selected_apartment:
#         recommendation_df = recommend_properties_with_scores(st.session_state.selected_apartment)
#         st.dataframe(recommendation_df)


import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="Apartment Recommendations", page_icon="🏢")
st.title("🏢 Apartment Recommendations")

# --------------------------- LOAD DATA ---------------------------

location_df = pickle.load(open('datasets/location_distance.pkl', 'rb'))
cosine_sim1 = pickle.load(open('datasets/cosine_sim1.pkl', 'rb'))
cosine_sim2 = pickle.load(open('datasets/cosine_sim2.pkl', 'rb'))
cosine_sim3 = pickle.load(open('datasets/cosine_sim3.pkl', 'rb'))

# --------------------------- SESSION STATE ---------------------------

for key in ['search_clicked', 'results', 'selected_apartment']:
    if key not in st.session_state:
        st.session_state[key] = None if key == 'selected_apartment' else False if key == 'search_clicked' else None

# --------------------------- HEADER ---------------------------

def show_header():
    st.markdown("""
        <style>
            .recommend-title {
                font-size: 36px;
                font-weight: bold;
                color: #2C3E50;
                text-align: center;
                margin-bottom: 40px;
            }
        </style>
        <div class="recommend-title">🏘️ Apartment Recommender</div>
    """, unsafe_allow_html=True)

# --------------------------- CORE LOGIC ---------------------------

def recommend_properties_with_scores(property_name, top_n=5):
    cosine_sim_matrix = 0.5 * cosine_sim1 + 0.8 * cosine_sim2 + 1.0 * cosine_sim3
    sim_scores = list(enumerate(cosine_sim_matrix[location_df.index.get_loc(property_name)]))
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

    top_indices = [i[0] for i in sorted_scores]
    top_scores = [i[1] for i in sorted_scores]
    top_properties = location_df.index[top_indices].tolist()

    return pd.DataFrame({
        '🏢 Property Name': top_properties,
        '⭐ Similarity Score': [round(score, 3) for score in top_scores]
    })

# --------------------------- FORM UI ---------------------------

def render_search_form():
    st.subheader("📍 Select Location and Radius")

    selected_location = st.selectbox('Choose a location:', sorted(location_df.columns.tolist()))
    radius = st.number_input('Radius (in km)', min_value=0.0, step=0.5)

    if st.button('Search Apartments'):
        result_ser = location_df[location_df[selected_location] <= radius * 1000][selected_location].sort_values()
        if result_ser.empty:
            st.warning("⚠️ No apartments found within the given radius.")
            st.session_state.search_clicked = False
        else:
            st.session_state.search_clicked = True
            st.session_state.results = result_ser
            st.session_state.selected_apartment = list(result_ser.index)[0]

# --------------------------- RADIO + RECOMMEND ---------------------------

def show_recommendation_interface():
    result_ser = st.session_state.results
    apartment_list = list(result_ser.index)
    distance_labels = [f"{round(d / 1000, 2)} km" for d in result_ser.values]

    selected = st.radio(
        "🏢 Select an apartment to get recommendations",
        apartment_list,
        captions=distance_labels,
        index=apartment_list.index(st.session_state.selected_apartment)
    )
    st.session_state.selected_apartment = selected

    if selected:
        st.subheader("🏡 Top Recommended Properties")
        df = recommend_properties_with_scores(selected)
        st.dataframe(df, use_container_width=True)

# --------------------------- MAIN APP ---------------------------

show_header()
render_search_form()
if st.session_state.search_clicked and st.session_state.results is not None:
    show_recommendation_interface()
