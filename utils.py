#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy import linalg
from collections import defaultdict
import docx


# reading text from docx
def getText(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)


def softmax(z):
    '''
    Inputs: 
        z: output scores from the hidden layer
    Outputs: 
        yhat: prediction (estimate of y)
    '''
    ### START CODE HERE (Replace instances of 'None' with your own code) ###
    # Calculate yhat (softmax)
    yhat = np.exp(z) / np.sum(np.exp(z), axis=0)
    ### END CODE HERE ###
    return yhat


def relu(z):
    # relu functon
    return np.maximum(z, 0)


def get_idx(words, word2Ind):
    idx = []
    for word in words:
        idx = idx + [word2Ind[word]]
    return idx


def pack_idx_with_frequency(context_words, word2Ind):
    freq_dict = defaultdict(int)
    for word in context_words:
        freq_dict[word] += 1
    idxs = get_idx(context_words, word2Ind)
    packed = []
    for i in range(len(idxs)):
        idx = idxs[i]
        freq = freq_dict[context_words[i]]
        packed.append((idx, freq))
    return packed


def get_vectors(data, word2Ind, V, C):
    i = C
    while True:
        y = np.zeros(V)
        x = np.zeros(V)
        center_word = data[i]
        # one-hot vector for center word
        y[word2Ind[center_word]] = 1
        
        context_words = data[(i - C) : i] + data[(i + 1) : (i + C + 1)]
        num_ctx_words = len(context_words)
        for idx, freq in pack_idx_with_frequency(context_words, word2Ind):
            # one-hot vector for context words
            x[idx] = freq / num_ctx_words
        yield x, y
        # sliding window to the right
        i += 1
        if i >= len(data):
            print("i is being set to 0")
            i = 0


def get_batches(data, word2Ind, V, C, batch_size):
    batch_x = []
    batch_y = []
    for x, y in get_vectors(data, word2Ind, V, C):
        while len(batch_x) < batch_size:
            batch_x.append(x)
            batch_y.append(y)
        else:
            yield np.array(batch_x).T, np.array(batch_y).T
            batch_x = []
            batch_y = []
            

def compute_pca(data, n_components=2):
    """
    Input: 
        data: of dimension (m,n) where each row corresponds to a word vector
        n_components: Number of components you want to keep.
    Output: 
        X_reduced: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """

    m, n = data.shape

    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = linalg.eigh(R)
    # sort eigenvalue in decreasing order
    # this returns the corresponding indices of evals and evecs
    idx = np.argsort(evals)[::-1]

    evecs = evecs[:, idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :n_components]
    ### END CODE HERE ###
    return np.dot(evecs.T, data.T).T


def get_dict(data):
    """
    Input:
        data: the data you want to pull from
    Output:
        word2Ind: returns dictionary mapping the word to its index
        Ind2Word: returns dictionary mapping the index to its word
    """
    
    # words = nltk.word_tokenize(data)
    words = sorted(list(set(data)))
    n = len(words)
    idx = 0
    # return these correctly
    word2Ind = {}
    Ind2word = {}
    for k in words:
        word2Ind[k] = idx
        Ind2word[idx] = k
        idx += 1
    return word2Ind, Ind2word


def euclidean_distance(v, w):
    d = linalg.norm(v-w)
    return d

def cosine_similarity(v, w):
    c = np.dot(v,w) / (linalg.norm(v)*linalg.norm(w))
    return c


# Define a function to find K closest words to a vector:
def find_closest_word(word, k, embeddings):
    
    most_closest_words = []
    word_emb = embeddings[word]
    similar_word = ''
    
    for w in embeddings.keys():
        if word != w:
            # get the word embedding
            w_emb = embeddings[w]
            # calculating cosine similarity
            cur_similarity = cosine_similarity(word_emb, w_emb)
            # store the similar_word as a tuple, which contains the word and the similarity
            similar_word = (w, cur_similarity)
            # append each tuple to list
            most_closest_words.append(similar_word)
    # sort based on more similarity
    most_closest_words.sort(key=lambda y: -y[1])
    return most_closest_words[:k]



