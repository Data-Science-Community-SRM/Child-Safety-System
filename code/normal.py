import pandas as pd
import numpy as np

norm=pd.read_csv('/content/noraml_nodups.csv')
norm.head()

norm.drop_duplicates(subset='text',inplace=True)
norm.shape

!unzip '/content/archive (1).zip'
ham=pd.read_csv('/content/spam.csv',encoding='ISO-8859-1')
ham.head()
ham.drop('Unnamed: 2',axis=1,inplace=True)
ham.drop('Unnamed: 3',axis=1,inplace=True)
ham.drop('Unnamed: 4',axis=1,inplace=True)
ham.rename(columns={'v1':'labels','v2':'text'},inplace=True)
norm3=ham[ham.columns[[1,0]]]
norm3.head()
norm4=norm3[norm3.labels!='spam']
norm4.drop_duplicates(subset='text',inplace=True)
norm4.isna().sum()
norm4['labels']=norm4['labels'].replace(['ham'],0)
stuff=[norm,norm4]
normal_texts=pd.concat(stuff)
normal_texts.head()
normal_texts.dropna(axis=0,inplace=True)
normal_texts.shape
normal_texts.to_csv('normal_texts_final.csv')
