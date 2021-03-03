import streamlit as st
import pandas as pd
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer 
import json
#get_text is a simple function to get user input from text_input
def get_text():
    input_text = st.text_input("You: ","So, what's in your mind")
    return input_text
#data input
# data = json.loads(open(r'./data/data_tolokers.json','r').read())#change path accordingly
# data2 = json.loads(open(r'./data/summer_wild_evaluation_dialogs.json','r').read())#change path accordingly
# tra = []
# for k, row in enumerate(data):
#     #print(k)
#     tra.append(row['dialog'][0]['text'])
# for k, row in enumerate(data2):
#     #print(k)
#     tra.append(row['dialog'][0]['text'])
df= pd.read_csv('final_texts.csv')
tra = list(df['texts'])
st.sidebar.title("NLP Bot")
st.title("""
NLP Bot  
NLP Bot is an NLP conversational chatterbot. Initialize the bot by clicking the "Initialize bot" button. 
""")
bot = ChatBot(name = 'PyBot', read_only = False,preprocessors=['chatterbot.preprocessors.clean_whitespace','chatterbot.preprocessors.convert_to_ascii','chatterbot.preprocessors.unescape_html'], logic_adapters = ['chatterbot.logic.MathematicalEvaluation','chatterbot.logic.BestMatch'])
ind = 1
if st.sidebar.button('Initialize bot'):
    trainer2 = ListTrainer(bot) 
    trainer2.train(tra)
    st.title("Your bot is ready to talk to you")
    ind = ind +1
        
user_input = get_text()
if True:
    st.text_area("Bot:", value=bot.get_response(user_input), height=200, max_chars=None, key=None)
else:
    st.text_area("Bot:", value="Please start the bot by clicking sidebar button", height=200, max_chars=None, key=None)
