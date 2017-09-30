
# coding: utf-8


# In[6]:

import numpy as np
import pandas as pd


# ## The Data
# 
# **Read the yelp.csv file and set it as a dataframe called yelp.**

# In[7]:

yelp = pd.read_csv('yelp.csv')



# In[96]:

yelp.head()


# In[97]:

yelp.info()


# In[99]:

yelp.describe()


# In[8]:

yelp['text length'] = yelp['text'].apply(len)


# # EDA
# 

# In[9]:

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
get_ipython().magic('matplotlib inline')



# In[10]:

g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length')



# In[11]:

sns.boxplot(x='stars',y='text length',data=yelp,palette='rainbow')



# In[12]:

sns.countplot(x='stars',data=yelp,palette='rainbow')



# In[13]:

stars = yelp.groupby('stars').mean()
stars



# In[14]:

stars.corr()



# In[15]:

sns.heatmap(stars.corr(),cmap='coolwarm',annot=True)


# ## NLP Classification Task

# In[16]:

yelp_class = yelp[(yelp.stars==1) | (yelp.stars==5)]



# In[17]:

X = yelp_class['text']
y = yelp_class['stars']



# In[18]:

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()



# In[19]:

X = cv.fit_transform(X)


# ## Train Test Split

# In[20]:

from sklearn.model_selection import train_test_split


# In[21]:

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)


# ## Training a Model
# 

# In[22]:

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()



# In[23]:

nb.fit(X_train,y_train)


# ## Predictions and Evaluations
# 

# In[24]:

predictions = nb.predict(X_test)



# In[25]:

from sklearn.metrics import confusion_matrix,classification_report


# In[26]:

print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))



# In[27]:

from sklearn.feature_extraction.text import  TfidfTransformer



# In[28]:

from sklearn.pipeline import Pipeline


# In[29]:

pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])



# ### Train Test Split

# In[36]:

X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)



# In[37]:


pipeline.fit(X_train,y_train)


# ### Predictions and Evaluation

# In[38]:

predictions = pipeline.predict(X_test)


# In[39]:

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


#Tf-Idf actually made things worse
