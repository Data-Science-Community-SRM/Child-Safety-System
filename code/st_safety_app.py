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
import random
from sklearn.model_selection import train_test_split

df=pd.read_csv('data/dataframe_texts.csv')

x=df['texts']
y=df['labels']
x=np.array(x)
y=np.array(y)
print(len(x))
print(len(y))

xtrain,x1,ytrain,y1=train_test_split(x,y,test_size=0.3,random_state=42)
xtest,xval,ytest,yval=train_test_split(x1,y1,test_size=0.1,random_state=42)
tokenizer = Tokenizer(filters='!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(xtrain.astype('str'))

html = """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#33ccff, #ccff33);
    color: white;
}
</style>
"""

st.markdown(html, unsafe_allow_html=True)

menu = ['Welcome', 'A chat based warning system']
with st.sidebar.beta_expander("Menu", expanded=False):
    option = st.selectbox('Choose: ', menu)
    st.subheader("Made with ❤️ by Data Science Community SRM")
#st.sidebar.beta_expander("Menu", expanded=False)
#option = st.selectbox('Choose', menu)
if option=='Welcome':
    st.header('Welcome to the child safety system.')
    st.image('data/predators.jpg')
    st.write('This application is meant to identify potential predators in chatrooms and warn the other user against such suspiscious behaviour.')
    st.write('The working application can be found in the drop-down menu.')
elif option=='A chat based warning system':
    st.header('A warning system that identifies potential predators and warns users against chatting with them.')
    st.write('You may begin a conversation and as it progresses, messages will be displayed by the bot to provide warnings or conclude it to be safe.')
    def preproc(text):
        x1=[]
        xlm=[]
        for i in range(len(text)):
            x1.append(''.join([word.lower() for word in text[i] if word not in string.punctuation]))
            lemma=WordNetLemmatizer()
            tokens=word_tokenize(x1[i])
            xlm.append(' '.join(lemma.lemmatize(word) for word in tokens))
        xlm=' '.join(xlm)
        st.write(xlm)
        return xlm
    def preds(l1):
        tok1=tokenizer.texts_to_sequences(l1)
        pad1=pad_sequences(tok1,maxlen=70)
        return pad1
    def chat():
        m=pd.read_csv('data/response.csv')
        inp=st.text_input("SAF: Hey there! Let's begin our convo :) ")
        inp=inp.lower()
        x=random.choice(m['0'])
        #txt=preproc(inp)
        st.write('SAF: ',x)
        model=tf.keras.models.load_model('data\lstm_glove_model.h5')
        tok=tokenizer.texts_to_sequences(x)
        pad=pad_sequences(tok,maxlen=70)
        pad1=preds(inp)
        res1=model.predict(pad)
        res2=model.predict(pad)
        ind1=np.argmax(res1)
        ind2=np.argmax(res2)
        #st.write(res1[ind1])
        #st.write(res2[ind2])
        if res1[ind1]>0.8 or res2[ind2]>0.8:
            st.write('Dangerous',height=200)
        else:
            st.write('OK',height=200)
    chat()
