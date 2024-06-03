#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import google.generativeai as genai
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
import neattext.functions as nfx
import neattext as nt
from nltk.stem import WordNetLemmatizer
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import Chroma
import pandas as pd


# In[2]:


os.environ['GOOGLE_API_KEY'] = "AIzaSyCLy24Ahfmeqw7hIfki1hiuO3a1b8hktms"
genai.configure(api_key = os.environ['GOOGLE_API_KEY'])


# In[3]:


data_1 = pd.read_csv(r"C:\Users\User\Downloads\Dataset__problem_1_.csv")


# In[4]:


data_1.head(4)


# In[5]:


first_column_name = data_1.columns[0]
data_1.drop(first_column_name,axis=1,inplace=True)


# In[6]:


data_1.head(4)


# In[7]:


data_1.dropna(inplace=True)


# In[8]:


data_1.isna().sum()


# In[9]:


data_1 = data_1.dropna(subset=['text'])


# In[10]:


data_1['text'] = data_1['text'].apply(lambda x: str(x) if not pd.isna(x) else '')


# In[11]:


sentences = data_1['text'].tolist()


# In[12]:


sentiment = data_1['sentiment'].tolist()


# In[13]:


embeddings_model = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')


# In[14]:


embeddings = embeddings_model.embed_documents(sentences)


# In[15]:


y = sentiment


# In[16]:


X = embeddings


# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state=99)


# In[18]:


from sklearn.ensemble import RandomForestClassifier
# create model variables
model= RandomForestClassifier()
# fit the model on smoted trainign data
result= model.fit(X, y)
# make predictions
y_pred= model.predict(X_test)


# In[19]:


# Get and reshape confusion matrix data
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib import pyplot as plt
matrix = confusion_matrix(y_test, y_pred)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(10,7))
sns.set(font_scale=1)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




