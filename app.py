import streamlit as st
import pickle
from closest_neighbours_page import show_closest_neighbours_page


def load_model():
    with open('embeddings.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

word_embeddings = data['embeddings']


st.sidebar.selectbox("Word Map Or Neighbours", ("Word Map","Closest Neighbours"))

show_closest_neighbours_page(word_embeddings)