#!/usr/bin/env python
# coding: utf-8

# # Import packages

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, f1_score, accuracy_score, recall_score, precision_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
#import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn.manifold import TSNE
#import matplotlib.patches as mpatches
import pickle


# # Read data

# In[2]:


import sklearn
print(sklearn.__version__)


# In[3]:


import sys
print(sys.version)


# In[4]:


df_data = pd.read_csv("adult.csv")


# In[5]:


## df_data.head()


# In[6]:


#df_data["workclass"].unique()


# # Check data

# In[7]:


#df_data.dtypes


# In[8]:


#df_data["nativecountry"].value_counts().index[0]


# In[9]:


#df_data["income"].value_counts()


# In[10]:


df_data = df_data.drop(['fnlwgt', 'educationalnum'], axis = 1)


# In[11]:


col_names = df_data.columns


# # Check Nulls and replace values

# In[12]:


for c in col_names:
    df_data = df_data.replace("?", np.NaN)
df_data = df_data.apply(lambda x:x.fillna(x.value_counts().index[0]))


# # Encoding

# In[13]:


category_col =['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry', 'income']
lbl_data = df_data.copy()
df_input = df_data[category_col].copy()
enc = preprocessing.LabelEncoder()
encoder_dict = dict()
for cat in category_col:
    df_input[cat] = df_input[cat].str.lstrip()
    enc = enc.fit(list(df_input[cat]) + ['Unknown'])
    encoder_dict[cat] = [cat for cat in enc.classes_]
    lbl_data[cat] = enc.transform(df_input[cat])


# In[14]:


##lbl_data.head()


# In[15]:


##print(encoder_dict)


# # Save Label Encoder

# In[16]:


encoder_pickle_out = open("encoder.pkl", "wb")
pickle.dump(encoder_dict, encoder_pickle_out)
encoder_pickle_out.close()


# # Label Data

# In[19]:


X = lbl_data.drop('income', axis = 1)
Y = lbl_data['income']


# # Split into train and test

# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)


# # Train & Test Data using Random Forest

# In[23]:


#rf = RandomForestClassifier(n_estimators=100, random_state=0)
#rf.fit(X_train, y_train)
#ypred = rf.predict(X_test)
#print(confusion_matrix(y_test, ypred))
#print(classification_report(y_test, ypred))
#print("Accuracy Score:", accuracy_score(y_test, ypred))
#print("Recall Score:", recall_score(y_test, ypred))
#print("Precision Score:", precision_score(y_test, ypred))
#print("ROC AUC Score: ", roc_auc_score(y_test, ypred))
#rf_fp, rf_tp, rf_threshold = roc_curve(y_test, ypred)
#print("Threshold:", rf_threshold)


# # Train & Test Data using Gradient Boosting Classifier

# In[24]:


gbc = GradientBoostingClassifier(n_estimators=100, random_state=0)
gbc.fit(X_train, y_train)
#ypred = gbc.predict(X_test)
#print(confusion_matrix(y_test, ypred))
#print(classification_report(y_test, ypred))
#print("Accuracy Score:", accuracy_score(y_test, ypred))
#print("Recall Score:", recall_score(y_test, ypred))
#print("Precision Score:", precision_score(y_test, ypred))
#print("ROC AUC Score: ", roc_auc_score(y_test, ypred))
#gbc_fp, gbc_tp, gbc_threshold = roc_curve(y_test, ypred)
#print("Threshold:", gbc_threshold)


# # Train & Test Data using Ada Boost Classifier

# In[25]:


#abc = AdaBoostClassifier(n_estimators=100, random_state=0)
#abc.fit(X_train, y_train)
#ypred = abc.predict(X_test)
#print(confusion_matrix(y_test, ypred))
#print(classification_report(y_test, ypred))
#print("Accuracy Score:", accuracy_score(y_test, ypred))
#print("Recall Score:", recall_score(y_test, ypred))
#print("Precision Score:", precision_score(y_test, ypred))
#print("ROC AUC Score: ", roc_auc_score(y_test, ypred))
#abc_fp, abc_tp, abc_threshold = roc_curve(y_test, ypred)
#print("Threshold:", abc_threshold)


# # ROC Curve

# In[26]:


#plt.figure(figsize=(20,10))
#plt.plot([0, 1], [0, 1], linestyle = "--")
#plt.plot(rf_fp, rf_tp, color="red", label = "Random Forest")
#plt.plot(gbc_fp, gbc_tp, color="green", label = "Gradient Booting")
#plt.plot(abc_fp, abc_tp, color="blue", label = "Ada Boosting")
#plt.xlabel("False Positive Rate")
#plt.ylabel("True Positive Rate")
#plt.title("ROC Curve")
#plt.legend()


# # Get most important features and their contribution in model

# In[27]:


#feature_importance_df = pd.DataFrame(X_train.columns, columns=["Feature"])
#feature_importance_df["Importance"] = rf.feature_importances_
#feature_importance_df.sort_values('Importance', ascending=False, inplace=True)
#feature_importance_df = feature_importance_df.head(20)
#feature_importance_df


# In[28]:


#plt.figure(figsize=(15,5))
#ax = feature_importance_df['Feature']
#plt.bar(range(feature_importance_df.shape[0]), feature_importance_df['Importance']*100)
#plt.xticks(range(feature_importance_df.shape[0]), feature_importance_df['Feature'], rotation = 20)
#plt.xlabel("Features")
#plt.ylabel("Importance")
#plt.title("Plot Feature Importances")


# # Save Model

# In[29]:


pickle_out = open("model.pkl", "wb")
pickle.dump(gbc, pickle_out)
pickle_out.close()


# # Test Model

# In[30]:


#pkl_file = open('encoder.pkl', 'rb')
#encoder_dict = pickle.load(pkl_file)
#pkl_file.close()


# In[31]:


#print(encoder_dict)


# In[32]:


#data = {'age': 27, 'workclass': 'Private', 'education': 'Bachelors', 'maritalstatus': 'Never-married', 'occupation': 'Sales', 'relationship': 'Husband', 'race': 'Other', 'gender': 'Female', 'capitalgain': 50000, 'capitalloss': 45, 'hoursperweek': 40, 'nativecountry': 'India'}
#print(data)
#df=pd.DataFrame([list(data.values())], columns=['age','workclass','education','maritalstatus','occupation','relationship','race','gender','capitalgain','capitalloss','hoursperweek','nativecountry'])


# In[33]:


##df.head()
#df.columns


# In[34]:


##category_col =['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry', 'income']
#category_col =['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']

#for cat in encoder_dict:

#for cat in encoder_dict:
#    for col in category_col:
#        le = preprocessing.LabelEncoder()
#        if cat == col:
#            le.classes_ = encoder_dict[cat]
#            for unique_item in df[col].unique():
#                if unique_item not in le.classes_:
#                    df[col] = ['Unknown' if x == unique_item else x for x in df[col]]
#            df[col] = le.transform(df[col])


# In[35]:


# Example data, assuming 'data' dictionary has been defined as in the previous snippet
#data = {'age': 27, 'workclass': 'Private', 'education': 'Bachelors', 'maritalstatus': 'Never-married', 'occupation': 'Sales', 'relationship': 'Husband', 'race': 'Other', 'gender': 'Female', 'capitalgain': 50000, 'capitalloss': 45, 'hoursperweek': 40, 'nativecountry': 'India'}

# Convert the data into a DataFrame for easier manipulation
#df = pd.DataFrame([data])

# Load the encoder dictionary from a pickle file
#with open('encoder.pkl', 'rb') as pkl_file:
#    encoder_dict = pickle.load(pkl_file)

# Preprocess and encode the categorical columns
#category_col = ['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']

#for col in category_col:
    # Initialize a LabelEncoder
#    le = preprocessing.LabelEncoder()

     # Convert the loaded classes list to a numpy array with dtype 'object'
#    le.classes_ = np.array(encoder_dict[col], dtype=object)

    # Check and replace unknown values with 'Unknown' (ensure 'Unknown' is handled correctly)
#    df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')

    # If 'Unknown' is not in le.classes_, you need to add it or handle this situation differently
#    if 'Unknown' not in le.classes_:
        # This line is just for illustration; handling needs to be decided based on your use case
#        print(f"'Unknown' category not handled for {col}. Please adjust.")
#        continue

    # Now transform the data
#    df[col] = le.transform(df[col])


# In[36]:


#model = pickle.load(open('model.pkl', 'rb'))
#features_list = df.values.tolist()
#prediction = model.predict(features_list)
#prediction = gbc.predict(features_list)
#print(prediction[0])


# In[ ]:
