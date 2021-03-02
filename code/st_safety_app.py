import pandas as pd
import numpy as np
import string
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import keras.backend as K
import streamlit as st


menu = ['Welcome', 'Chatbot warning system']
st.sidebar.beta_expander("Menu", expanded=False):
option = st.selectbox('Choose', menu)
if option=='Welcome':
    st.header('Welcome to the child safety system.')
elif option=='A chat based warning system':
    st.header('Chat safety')
    r1,r2=chat()
    if r1>0.8 or r2>0.8:
        print('This conversation maybe dangerous, beware!')
    else:
        break

def chat():
    u1,u2=[],[]
    i=2
    while i>0:
        inp1=input('User 1: ')
        inp2=input('User 2: ')
        u1.append(inp1)
        u2.append(inp2)
        i=i-1
    u1=preproc(u1)
    u2=preproc(u2)
    x,y=preds(u1,u2)
    o=np.argmax(x)
    k=np.argmax(y)
    m=x[o]
    n=y[k]
    return m,n

def preproc(text):
    x1=[]
    xlm=[]
    for i in range(len(u1)):
        x1.append(''.join([word.lower() for word in text[i] if word not in string.punctuation]))
        lemma=WordNetLemmatizer()
        tokens=word_tokenize(x1[i])
        xlm.append(' '.join(lemma.lemmatize(word) for word in tokens))
    return xlm

def preds(l1,l2):
    model=tf.keras.models.load_model('data\lstm_glove_model.h5')
    tok1=tokenizer.texts_to_sequences(l1)
    pad1=pad_sequences(tok1,maxlen=70)
    res1=model.predict(pad1)
    tok2=tokenizer.texts_to_sequences(l2)
    pad2=pad_sequences(tok2,maxlen=70)
    res2=model.predict(pad2)
    return res1,res2
