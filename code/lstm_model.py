from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Embedding, Conv2D, Concatenate, Flatten, Add, Dropout
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.losses import BinaryCrossentropy
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
import keras.backend as K

pd.read_csv('df1.csv')
df1.drop_duplicates(subset='texts',inplace=True)
df1.shape
x=df1['texts']
y=df1['labels']
x=np.array(x)
y=np.array(y)
#print(len(x))
#print(len(y))

xtrain,x1,ytrain,y1=train_test_split(x,y,test_size=0.3,random_state=42)
xtest,xval,ytest,yval=train_test_split(x1,y1,test_size=0.1,random_state=42)

tokenizer = Tokenizer(filters='!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(xtrain)

vocab_size=len(tokenizer.word_index)+1
print(vocab_size)
#texts to sequences
train_tok=tokenizer.texts_to_sequences(xtrain)
val_tok=tokenizer.texts_to_sequences(xval)
test_tok=tokenizer.texts_to_sequences(xtest)
#padding
train_pad=pad_sequences(train_tok,maxlen=50,padding='pre')
val_pad=pad_sequences(val_tok,maxlen=50,padding='pre')
test_pad=pad_sequences(test_tok,maxlen=50,padding='pre')

#model
model=Sequential()
model.add(Embedding(input_dim=vocab_size,output_dim=64,input_length=train_pad.shape[1],trainable=False,mask_zero=True))
model.add(LSTM(128,activation='tanh'))
model.add(Dropout(0.35))
model.add(Dense(512,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))

#model.summary()

ytrain=np.array(ytrain.astype('int'))
train_pad=np.array(train_pad)
yval=np.array(yval.astype('int'))
val_pad=np.array(val_pad)

red=ReduceLROnPlateau(monitor='val_accuracy',patience=3)
model.compile(optimizer=Adam(learning_rate=0.01),metrics=['accuracy'],loss='binary_crossentropy')
model.fit(x=train_pad,y=ytrain,batch_size=64,epochs=30,callbacks=red,validation_data=(val_pad,yval),validation_batch_size=64)
#val accuracy=87.37%
#model.save('texts_lstm_model.h5')
