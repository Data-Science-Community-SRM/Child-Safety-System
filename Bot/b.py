from flask import Flask, render_template, request
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.trainers import ListTrainer
#from keras.models import load_model
import keras
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer
from tensorflow.python.keras.preprocessing import text
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import streamlit as st
#app = Flask(__name__)

def init_bot():
    bot = ChatBot('ChatterBot',storage_adapter="chatterbot.storage.SQLStorageAdapter")
    trainer = ChatterBotCorpusTrainer(bot)
    trainer.train("chatterbot.corpus.english")
    return bot
#@app.route("/")
#def home():
#    return render_template("home.html")

#@app.route("/get")
def get_bot_response():
#    userText = request.args.get('msg')
    return str(bot.get_response(userText))

#if __name__ == "__main__":
#    app.run()
def load_model():
    model = keras.models.load_model('model.h5')
    model.summary()
    return model

def predict(imp):
    tokenizer = Tokenizer(filters='!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(imp)
    imp = tokenizer.texts_to_sequences(imp)
    imp = pad_sequences(imp,maxlen=50,padding='pre')
    #np.array.astype(int)
    a = model.predict(imp)
    res =np.argmax(a.round())
    print(res)
    return res 

bot =init_bot()
model= load_model()
st.title('Hey, Lets talk' )
text = st.text_input('Type you message')
st.sidebar.title('Status of Chat')
if st.button('send'):
    result = bot.get_response(text)
    st.text_area("Bot:",value=result)
    if predict(text)==1:
        st.sidebar.text_area("Status","Suspicious Behavior by user")
    elif predict(result)==1:
        st.sidebar.text_area("Status","Suspicious Behavior by Bot")
    else:
        st.sidebar.text_area("Status","SAFE")

 
        

