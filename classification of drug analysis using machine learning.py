#!/usr/bin/env python
# coding: utf-8

# import pandas as pd
# import numpy as np
# 
# from sklearn.svm import SVC #support vector classifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_iris
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report
# 
# 
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# %matplotlib inline
# 
# 
# from sklearn.metrics import accuracy_score
# 

# In[ ]:


training_data =pd.read_csv('drugtrial.csv')
training_data.head()


# In[ ]:


training_data['urlDrugName'].value_counts().head(15).plot(kind='bar')
plt.rcParams['figure.figsize']=(10,7)
plt.show()


# In[ ]:


training_data['rating'].value_counts().head(15).plot(kind='bar')
plt.xlabel('ratings')
plt.show()


# In[ ]:


training_data['effectiveness'].value_counts().plot(kind='pie')
plt.show()


# In[ ]:


training_data['condition'].value_counts().head(15).plot(kind='bar')
plt.xlabel('condition')
plt.show()


# In[ ]:


target=training_data.pop('sideEffects')
training_data.head()


# In[ ]:


cols=['urlDrugName','effectiveness','condition']
for x in cols:
    training_data[x]=pd.factorize(training_data[x])[0]
    
target=pd.factorize(target)[0]
training_data.head()


# In[ ]:


scaler=StandardScaler()
training_data=scaler.fit_transform(training_data)
training_data


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(training_data,target, test_size=0.2, random_state=0)


# In[ ]:


svm_clf=SVC().fit(x_train,y_train)

svm_pred=svm_clf.predict(x_test)

print(classification_report(y_test,svm_pred))


# In[ ]:


KNeighborsClassifier_clf=KNeighborsClassifier().fit(x_train,y_train)

KNeighborsClassifier_pred=KNeighborsClassifier_clf.predict(x_test)

print(classification_report(y_test,KNeighborsClassifier_pred))


# In[ ]:


LogisticRegression_clf=LogisticRegression().fit(x_train,y_train)
LogisticRegression_pred=LogisticRegression_clf.predict(x_test)

print(classification_report(y_test,LogisticRegression_pred))


# In[ ]:


rf_clf=RandomForestClassifier().fit(x_train,y_train)

rf_pred=rf_clf.predict(x_test)

print(classification_report(y_test,rf_pred))


# In[ ]:


from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)
prediction=clf.predict(x_test)


# In[ ]:


iris = load_iris()
df,target = iris.data, iris.target
clf = tree.DecisionTreeClassifier()
clf = clf.fit(df, target)
tree.plot_tree(clf) 


# In[ ]:


accuracy_scores = np.zeros(5)

# Support Vector Classifier
clf = SVC().fit(x_train, y_train)
prediction = clf.predict(x_test)
accuracy_scores[0] = accuracy_score(y_test, prediction)*100
print('Support Vector Classifier accuracy: {}%'.format(accuracy_scores[0]))

# Logistic Regression
clf = LogisticRegression().fit(x_train, y_train)
prediction = clf.predict(x_test)
accuracy_scores[1] = accuracy_score(y_test, prediction)*100
print('Logistic Regression accuracy: {}%'.format(accuracy_scores[1]))

# K Nearest Neighbors
clf = KNeighborsClassifier().fit(x_train, y_train)
prediction = clf.predict(x_test)
accuracy_scores[2] = accuracy_score(y_test, prediction)*100
print('K Nearest Neighbors Classifier accuracy: {}%'.format(accuracy_scores[2]))

# Random Forest
clf = RandomForestClassifier().fit(x_train, y_train)
prediction = clf.predict(x_test)
accuracy_scores[3] = accuracy_score(y_test, prediction)*100
print('Random Forest Classifier accuracy: {}%'.format(accuracy_scores[3]))

#Decision Tree Classifier
clf = tree.DecisionTreeClassifier().fit(x_train, y_train)
prediction = clf.predict(x_test)
accuracy_scores[4] = accuracy_score(y_test, prediction)*100
print('Random Forest Classifier accuracy: {}%'.format(accuracy_scores[4]))


plt.figure(figsize=(12,8))
colors = cm.rainbow(np.linspace(0, 1, 4))
labels = ['Support Vector Classifier', 'Logsitic Regression', 'K Nearest Neighbors', 'Random Forest','Decision tree Classifier']
plt.bar(labels,
        accuracy_scores,
        color = colors)
plt.xlabel('Classifiers',fontsize=18)
plt.ylabel('Accuracy',fontsize=18)
plt.title('Accuracy of various algorithms',fontsize=20)
plt.xticks(rotation=45,fontsize=12)
plt.yticks(fontsize=12)


# In[ ]:




