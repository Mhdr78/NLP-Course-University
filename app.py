import streamlit as st
import pickle
from closest_neighbours_page import show_closest_neighbours_page
from wordsmap_page import show_wordsmap_page

def load_model(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

data_v = load_model('./output/saved_weights_v_pytorch.pkl')
data_u = load_model('./output/saved_weights_u_pytorch.pkl')
data_m = load_model('./output/saved_weights_m_pytorch.pkl')

data = {'V': data_v, 'U': data_u, "Mean": data_m}


page = st.sidebar.selectbox("Word Map Or Neighbours", ("Word Map","Closest Neighbours"))

if page == 'Word Map':
    show_wordsmap_page()
elif page == "Closest Neighbours":
    show_closest_neighbours_page(data)