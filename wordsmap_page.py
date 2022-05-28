import streamlit as st
import streamlit.components.v1 as components
import codecs


def show_wordsmap_page():
    # st.title("Word embeddings map.")
    # embed streamlit docs in a streamlit app
    tensorboard_file = codecs.open('tensorboard.html','r')
    page = tensorboard_file.read()
    st.title("Tensorboard projector")
    components.html(page, width=1300, height=1000, scrolling=True)