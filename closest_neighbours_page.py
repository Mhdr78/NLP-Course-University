import streamlit as st
import numpy as np
from numpy import linalg
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# def cosine_similarity(v, w):
#     c = np.dot(v,w) / (linalg.norm(v)*linalg.norm(w))
#     return c

def euclidean_distance(v, w):
    v = np.asarray(v)
    w = np.asarray(w)
    d = linalg.norm(v-w)
    return d

def find_closest_word(word, k, dist_type, embeddings):
    most_closest_words = []
    word_emb = embeddings[word]
    similar_word = ''
    
    for w in embeddings.keys():
        if w != word:
            # get the word embedding
            w_emb = embeddings[w]
            # calculating distance
            if dist_type == "Cosine":
                distance = cosine_similarity(np.asarray(word_emb).reshape(1,-1), np.asarray(w_emb).reshape(1,-1))
            elif dist_type == "Euclidean":
                distance = euclidean_distances(np.asarray(word_emb).reshape(1,-1), np.asarray(w_emb).reshape(1,-1))

            # store the similar_word as a tuple, which contains the word and the similarity
            similar_word = (w, distance)
            # append each tuple to list
            most_closest_words.append(similar_word)
    # sort based on more similarity
    if dist_type == "Cosine":
        most_closest_words.sort(key=lambda y: -y[1])
    else:
        most_closest_words.sort(key=lambda y: y[1])
    return most_closest_words[:k]


def show_closest_neighbours_page(data):
    st.title("Closest neighbours")
    st.write("""### Please choose a word from the vocabulary and select how many words you want to show""")
    
    weight_type = st.selectbox("Weight Type", ("V", "U", "Mean"))
    embeddings = data[weight_type]['embeddings']

    vocabulary = embeddings.keys()

    word = st.selectbox("Vocabulary", vocabulary)
    dist_type = st.selectbox("Distance Type", ("Cosine", "Euclidean"))
    

    k = st.slider("Number of words", 1, 30, 15)
    
    ok = st.button("Show result")

    if ok:
        st.subheader(f"Result is based on {dist_type}")
        result = find_closest_word(word, k, dist_type, embeddings)
        for res in result:
            st.write(f"{res[0]} : {res[1][0][0]:.4f}")
        