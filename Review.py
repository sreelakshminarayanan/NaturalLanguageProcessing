
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from textblob import TextBlob, Word
import nltk
from nltk.stem.snowball import SnowballStemmer
get_ipython().magic('matplotlib inline')


# In[3]:


review = pd.read_csv('Reviews.csv')


# In[4]:


review.head()


# In[5]:


review.describe()


# In[6]:


review.info()


# In[9]:


review.Score.plot.box(grid='True')


# In[19]:


pd.crosstab(review.HelpfulnessDenominator, review.Score)


# In[18]:


pd.crosstab(review.HelpfulnessNumerator, review.Score)


# In[5]:


review.tail()


# In[14]:


review.Score.value_counts().plot()


# In[6]:


review.describe()


# In[7]:


feature_cols = ['Text', 'HelpfulnessNumerator', 'HelpfulnessDenominator']
x = review[feature_cols]
y = review.Score


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)


# In[9]:


# use CountVectorizer with text column only
vect = CountVectorizer(stop_words='english')


# In[10]:


X_train_dtm = vect.fit_transform(X_train.Text)


# In[11]:


X_test_dtm = vect.transform(X_test.Text)


# In[12]:


# cast other feature columns to float and convert to a sparse matrix
extra = sp.sparse.csr_matrix(X_train.drop('Text', axis=1).astype(float))
extra.shape


# In[13]:


# combine sparse matrices
X_train_dtm_extra = sp.sparse.hstack((X_train_dtm, extra))
X_train_dtm_extra.shape


# In[14]:


# repeat for testing set
extra = sp.sparse.csr_matrix(X_test.drop('Text', axis=1).astype(float))
X_test_dtm_extra = sp.sparse.hstack((X_test_dtm, extra))
X_test_dtm_extra.shape


# In[15]:


# use logistic regression with text column only
logreg = LogisticRegression(C=1e9)
logreg.fit(X_train_dtm, y_train)
y_pred_class = logreg.predict(X_test_dtm)
print (metrics.accuracy_score(y_test, y_pred_class))


# In[16]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred_class)

