#!/usr/bin/env python
# coding: utf-8

# ## Credit Card Fraud Detection

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("creditcard.csv",sep=',')
df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.describe()


# ## Preprocessing the Data

# In[6]:


df.isnull().sum()


# In[7]:


df=df.fillna(method='ffill')


# In[8]:


df.isnull().sum()


# ## feature Scaling

# In[9]:


from sklearn.preprocessing import StandardScaler


# In[10]:


sc=StandardScaler()
df['Amount']=sc.fit_transform(pd.DataFrame(df['Amount']))


# In[11]:


df.head()


# In[12]:


df.shape


# In[13]:


# Check duplicate data
df.duplicated().sum()


# In[14]:


# Remove duplicate data
df=df.drop_duplicates()


# In[15]:


df.shape


# In[16]:


# Get Fraud and Normal datasets

fraud=df[df['Class']==1]
normal=df[df['Class']==0]


# In[17]:


print (fraud.shape,normal.shape)


# In[18]:


df['Class'].value_counts()


# In[19]:


sns.countplot(x='Class',data=df)


# In[20]:


corr=df.corr()
plt.figure(figsize=(30,40))
sns.heatmap(corr,annot=True,cmap='coolwarm')


# In[21]:


# store feature matrix in x and response in vector y
x=df.drop('Class',axis=1)
y=df['Class']


# ## Splitting the dataset into training and test set

# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[24]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[25]:


model.fit(x_train,y_train)


# In[26]:


y_pred = model.predict(x_test)


# In[27]:


from sklearn.metrics import precision_score,f1_score,accuracy_score


# In[28]:


accuracy_score(y_test,y_pred)


# In[29]:


precision_score(y_test,y_pred)


# In[30]:


f1_score(y_test,y_pred)


# In[31]:


from sklearn.ensemble import RandomForestClassifier


# In[32]:


model_rf1=RandomForestClassifier()


# In[33]:


model_rf1.fit(x_train,y_train)


# In[34]:


y_pred_rf1 = model.predict(x_test)


# In[35]:


accuracy_score(y_test,y_pred_rf1)


# In[36]:


precision_score(y_test,y_pred_rf1)


# In[37]:


f1_score(y_test,y_pred_rf1)


# ## Handling imbalance Dataset

# In[38]:


fraud=df[df['Class']==1]
normal=df[df['Class']==0]


# In[39]:


normal.shape,fraud.shape


# In[40]:


normal_sample=normal.sample(n=437)


# In[41]:


normal_sample=normal.sample(n=437)


# In[42]:


new_df=pd.concat([normal_sample,fraud],ignore_index=True)


# In[43]:


new_df['Class'].value_counts()


# In[44]:


new_df.head()


# In[45]:


x=new_df.drop('Class',axis=1)
y=new_df['Class']


# In[46]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.20,random_state=42)


# ## logistic regression and understading

# In[47]:


from sklearn.linear_model import LogisticRegression
model1=LogisticRegression()


# In[48]:


# Training the Logistic Regression model with training data
model1.fit(x_train,y_train)


# In[49]:


y_pred = model1.predict(x_test)


# In[50]:


from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score


# In[51]:


accuracy_score(y_test,y_pred)


# In[52]:


precision_score(y_test,y_pred)


# In[53]:


recall_score(y_test,y_pred)


# In[54]:


f1_score(y_test,y_pred)


# ## Random Forest Classifier after undersampling

# In[55]:


from sklearn.ensemble import RandomForestClassifier


# In[56]:


model_rf=RandomForestClassifier()


# In[57]:


model_rf.fit(x_train,y_train)


# In[58]:


y_pred_rf=model_rf.predict(x_test)


# In[59]:


accuracy_score(y_test,y_pred_rf)


# In[60]:


precision_score(y_test,y_pred_rf)


# In[61]:


recall_score(y_test,y_pred_rf)


# In[62]:


f1_score(y_test,y_pred_rf)


# In[63]:


final_result=pd.DataFrame({'Models':['Log.R','Rand.Fr'],
                           "Accuracy":[accuracy_score(y_test,y_pred)*100,
                                 accuracy_score(y_test,y_pred_rf)*100]})


# In[64]:


final_result


# In[65]:


sns.barplot(x='Models', y='Accuracy', data=final_result)


# ## handling imbalance data

# In[66]:


x=df.drop('Class',axis=1)
y=df['Class']


# In[67]:


x.shape,y.shape


# In[68]:


from imblearn.over_sampling import SMOTE


# In[69]:


x_res,y_res=SMOTE().fit_resample(x,y)


# In[70]:


y_res.value_counts()


# In[71]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.20,random_state=42)


# In[72]:


model1=LogisticRegression()
model1.fit(x_train,y_train)


# In[73]:


y_pred = model1.predict(x_test)


# In[74]:


accuracy_score(y_test,y_pred)


# In[75]:


precision_score(y_test,y_pred)


# In[76]:


recall_score(y_test,y_pred)


# In[77]:


f1_score(y_test,y_pred)


# In[78]:


from sklearn.ensemble import RandomForestClassifier


# In[79]:


model_rf=RandomForestClassifier()
model_rf.fit(x_train,y_train)


# In[80]:


y_pred_rf=model_rf.predict(x_test)


# In[81]:


accuracy_score(y_test,y_pred_rf)


# In[82]:


precision_score(y_test,y_pred_rf)


# In[83]:


recall_score(y_test,y_pred_rf)


# In[84]:


f1_score(y_test,y_pred_rf)


# In[85]:


final_result=pd.DataFrame({'Models':['Log.R','Rand.Fr'],"Accuracy":[accuracy_score(y_test,y_pred)*100,
accuracy_score(y_test,y_pred_rf)*100]})


# In[86]:


final_result


# In[87]:


sns.barplot(x='Models', y='Accuracy', data=final_result)

