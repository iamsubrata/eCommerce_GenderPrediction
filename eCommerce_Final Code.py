#!/usr/bin/env python
# coding: utf-8

# # Subrata Paul final code

# # Gender Prediction for E-Commerce
# With the evolution of the information and communication technologies and the rapid growth of the Internet for the exchange and distribution of information, Electronic Commerce (e-commerce) has gained massive momentum globally, and attracted more and more worldwide users overcoming the time constraints and distance barriers.
# 
# It is important to gain in-depth insights into e-commerce via data-driven analytics and identify the factors affecting product sales, the impact of characteristics of customers on their purchase habits.
# 
# It is quite useful to understand the demand, habits, concern, perception, and interest of customers from the clue of genders for e-commerce companies. 
# 
# However, the genders of users are in general unavailable in e-commerce platforms. To address this gap the aim here is to predict the gender of e-commerce’s participants from their product viewing records.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data_org=pd.read_csv('D:\\DS\\Janata Hack_AnalyticsVidhya\\train.csv')


# In[3]:


data_org.head()


# In[64]:


data_org.isnull().sum()


# In[4]:


#Checking the ratio of genders in the dataset
sns.countplot(data_org['gender'])


# In[5]:


data_org['gender'].value_counts()


# In[6]:


# Analysis of the Session Durations


# In[7]:


#using to_datetime function of pandas to convert the data related columns 
data_org['startTime']=pd.to_datetime(data_org['startTime'],format='%d/%m/%y %H:%M')
data_org['endTime']=pd.to_datetime(data_org['endTime'],format='%d/%m/%y %H:%M')


# In[8]:


data_org.info()


# In[9]:


#Calculating the duration or amount of time spect in the session in mins

data_org['Duration_in_min']=((data_org['endTime']-data_org['startTime']).dt.total_seconds())/60


# In[10]:


data_org.head(10)


# In[11]:


#Checking session duration analysis
data_org.hist(bins=50,figsize=(20,15))
plt.show()


# In[12]:


#Lets check for female how  duration is distributed
data_org[data_org['gender']=='female'].describe()


# In[13]:


#Above shows some max as weird value 49269 mins which means almost 34 days!!!! are you kidding me !!!!

data_max_time=data_org[data_org['Duration_in_min']==49269.000000].copy()
data_max_time


# In[14]:


##Yes the start and the end time shows session was open for more than a month !!..can be a outlier have to check the distribution


# In[15]:


#Checking what all products veiwed in such a long session !!
product_max=data_max_time['ProductList'].tolist()
type(product_max)
print(product_max)

As the result shows only 3-4 categories , might have kept the session open..or not sure
# In[16]:


#Lets check for male how  duration is distributed
data_org[data_org['gender']=='male'].describe()

Same observation as females , few males have kept sessions open for weirdly long time
# In[17]:


#Checking what all products veiwed in such a long session by a guy !!
male_max_time=data_org[data_org['Duration_in_min']==36982.000000].copy()
male_max_time


# In[18]:


#Checking the median values as average/mean is highly impacted by the outliers which is a loads in this datasets
sns.boxplot(y=data_org['gender'],x=data_org['Duration_in_min'])
plt.show()


# In[19]:


#Session more that 1 hour in a e commerce portal looks very unlikely
#Checking how many values are above 60 mins
data_org[data_org['Duration_in_min']>60].shape

Above shows duration more than 1 hour is 0.01% ....as expected very rare
# In[20]:


#Checking gender ratio for session dureation beyond 1 hr
data_org.loc[data_org['Duration_in_min']>60,'gender'].value_counts()


# In[21]:


#lets see the distribution for the data of duration till 1 hr
data_60=data_org[data_org['Duration_in_min']<=60].copy()
sns.boxplot(y=data_60['gender'],x=data_60['Duration_in_min'])
plt.show()


# In[22]:


data_60.describe()


# In[23]:


sns.distplot(data_60['Duration_in_min'])


# In[24]:


#Lets convert of the data where duration above 60 mins as 60  as 60 is a long enough time for a session to be open
data_ciel=data_org.copy()
data_ciel['Duration_in_min'].where(data_ciel['Duration_in_min']<60,60,inplace=True)


# In[25]:


data_ciel.describe()


# In[26]:


sns.boxplot(y=data_ciel['gender'],x=data_ciel['Duration_in_min'])
plt.show()


# In[27]:


data_ciel.hist(bins=30,figsize=(10,10))

Looks better in terms of duration distribution , Also we didnt any data as we accomodated the high duration data in 1 hour ceiling
# # Analysis of the ProductList

# In[28]:


#Checking the sample data
data_ciel.head()


# In[29]:


#female sample
data_ciel['ProductList'][0]


# In[30]:


#male Sample
data_ciel['ProductList'][1][0:6]

Going by intuition that first category is a good differentiator between the gender as male and female,so creating a new column as cat_1 for the parent category.
# In[31]:


cat_1=[]

for i in data_ciel['ProductList']:
    cat_1.append(i[0:6])


# In[32]:


cat_1[1]


# In[33]:


df_cat1=pd.DataFrame(list(cat_1),columns=['Cat_1'])


# In[34]:


df_cat1.head()


# In[35]:


df_cat1=pd.concat([data_ciel,df_cat1],axis=1)


# In[36]:


df_cat1.head()


# In[37]:


sns.countplot(df_cat1['Cat_1'],hue=df_cat1['gender'])
plt.xticks(rotation=90)

Above shows that our intuition of going with the parent category makes sense , as one of the categories approximately defines the genders
# In[38]:


#Trying same way with the sub category
cat_2=[]

for i in data_ciel['ProductList']:
    cat_2.append(i[7:13])


# In[39]:


df_cat2=pd.DataFrame(list(cat_2),columns=['Cat_2'])


# In[40]:


df_cat2.head()


# In[41]:


df_cat1_2=pd.concat([df_cat1,df_cat2],axis=1)


# In[42]:


df_cat1_2.head()


# In[43]:


plt.figure(figsize=(20,20))
sns.countplot(df_cat1_2['Cat_2'],hue=df_cat1_2['gender'])
plt.grid()
plt.xticks(rotation=90)

Above shows that this category also defines approximately , might be along with a combination of parent and sub category it will give better predictions, 
but looks like the sub category values are bit higher 30+ , handling this as categorical will not make much sense. So as of now we will 
go with the parent category , will aim for a simple model . Later on we can analyse further
# In[44]:


#Splitting dataset into the features & predicted/dependent variable
X=df_cat1[['Duration_in_min','Cat_1']]
y=df_cat1['gender']


# In[45]:


X.head()


# In[46]:


y.head()


# In[47]:


#Handling the categorical variables
X=pd.get_dummies(X)


# In[48]:


X.head()


# In[49]:


#Splitting the training and the test dataset, keeping(startifying) the dataset ratio as per the gender
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=df_cat1['gender'])


# In[50]:


from sklearn.ensemble import RandomForestClassifier


# In[51]:


#Using Random forest classifier as its good with handling overfitting
clf_1=RandomForestClassifier(n_estimators=500,random_state=42)


# In[52]:


clf_1.fit(X_train,y_train)


# In[53]:


y_pred=clf_1.predict(X_test)


# In[54]:


print(accuracy_score(y_test,y_pred))


# In[55]:


pd.crosstab(index=y_test,columns=y_pred,rownames=['Actual Gender'],colnames=['Predicted Gender'])


# # Testing the classifier with test data

# In[62]:


def gender_prediction():
    file_path=input('Enter test File path:')
    test_data=pd.read_csv(file_path)
    #using to_datetime function of pandas to convert the data related columns 
    test_data['startTime']=pd.to_datetime(test_data['startTime'],format='%d/%m/%y %H:%M')
    test_data['endTime']=pd.to_datetime(test_data['endTime'],format='%d/%m/%y %H:%M')
    #Calculating the duration or amount of time spect in the session in mins

    test_data['Duration_in_min']=((test_data['endTime']-test_data['startTime']).dt.total_seconds())/60
    test_ciel=test_data.copy()
    test_ciel['Duration_in_min'].where(test_ciel['Duration_in_min']<60,60,inplace=True)
    
    cat_1=[]
    for i in test_ciel['ProductList']:
        cat_1.append(i[0:6])
    d_cat1=pd.DataFrame(list(cat_1),columns=['Cat_1'])
    d_cat1=pd.concat([test_ciel,d_cat1],axis=1)
    X=d_cat1[['Duration_in_min','Cat_1']]
    ##y=d_cat1['gender']
    X=pd.get_dummies(X)
    y_pred=clf_1.predict(X)
    y_pred=pd.Series(y_pred)
    
    final_df=pd.concat([test_data[['session_id']],y_pred],axis=1)
    final_df.to_csv('Test_result.csv')


# In[63]:


gender_prediction()


# In[ ]:




