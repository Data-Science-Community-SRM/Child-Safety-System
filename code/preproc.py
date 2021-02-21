import json

df=pd.read_csv('Combined_data.csv')
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
