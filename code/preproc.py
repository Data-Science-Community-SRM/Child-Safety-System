import json
import pkg_resources
from tqdm import tqdm
import re
import pandas as pd
import pickle
from symspellpy.symspellpy import SymSpell,Verbosity
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#pip install symspellpy
#nltk.download('wordnet')
#nltk.download('punkt')

df=pd.read_csv('dataframe_texts.csv')
x,y=[],[]
x=df['text']
y=df['labels']
l=len(x)
x1=[]
#tokenize, remove punctuations and convert upper case letters to lower case
for i in range(l):
    x1.append(''.join([word.lower() for word in x[i] if word not in string.punctuation]))
#print(x1[1])
#lemmatize the words
lemma=WordNetLemmatizer()
xlm=[]
for i in range(l):
    tokens=word_tokenize(x1[i])
    xlm.append(' '.join(lemma.lemmatize(word) for word in tokens))
#remove contractions
l1=json.loads('contractions.json')
cont=list(l1.values())
print(type(l1))
y=[l1.keys()]
x=[l1.values()]
sntnc=[]
for i in range(len(xlm)):
    w=xlm[i].split()
    for a,b in l1.items():
        for j,k in enumerate(w):
            if k==a:
                w[j]=b.lower()
    sntnc.append(w)
i1=[]
for h in range(len(sntnc)):
    i1.append(' '.join(sntnc[h]))
x_train=[]
x_train=i1

with open('x_train.pkl','wb') as f:
    pickle.dump(x_train,f)

spellchk=SymSpell(max_dictionary_edit_distance=3,prefix_length=5)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")

spellchk.load_dictionary(dictionary_path,term_index=0,count_index=1)
spellchk.load_bigram_dictionary(dictionary_path,term_index=0,count_index=2)

normal=[]
for sent in tqdm(df['texts']):
    x=str(sent).split()
    for i in range(len(x)):
        w=x[i]
        if not w.isdigit() and not (w.lower() in spellchk.words.keys()):
            sug=spellchk.lookup(w,Verbosity.TOP,2)
            if len(sug)>0:
                corr=sug[0].term
                rep=corr
            else:
                rep=re.sub(r'([\w])\1+', r'\1', str(w))
            w=rep
            x[i]=w
    normal.append(' '.join(x).strip())
df['texts']=normal
df.to_csv('dataframe_texts.csv')
