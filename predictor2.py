import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

diabetesData= pd.read_csv("pima.csv")
print(diabetesData.shape)
diabetesData.head()
y = diabetesData['Outcome']
X = diabetesData.drop('Outcome', axis=1, inplace=False)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=['Pregnancies','Glucose','BloodPressure','SkinThickness',
                             'Insulin','BMI','DiabetesPedigreeFunction','Age'])
X.head()
# Models
# A
# 1. Decision Tree
# 2. KNN
# 3. Guassian Naive Bayes
# 4. Random Forest
# 5. SVM

# B
# 1. Voting Classifier


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# Decision Tree

from sklearn.tree import DecisionTreeClassifier
dectree = DecisionTreeClassifier()
dectree.fit(X_train,y_train)

y_pred_dectree = dectree.predict(X_test)

print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_dectree))
print('\n')
print('Accuracy')
print(accuracy_score(y_test, y_pred_dectree))

# KNN
from sklearn.neighbors import KNeighborsClassifier

error_rate = []

for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))

plt.plot(range(1, 20), error_rate, color='blue', linestyle='--', markersize=10, markerfacecolor='red', marker='o')

plt.title('K versus Error rate')

plt.xlabel('K')
plt.ylabel('Error rate')

# The accuracy on the test dataset is maximum with 13 neighbors

knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_knn))
print('\n')
print('Accuracy')
print(accuracy_score(y_test, y_pred_knn))

#GuassianNB

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred_gnb = gnb.predict(X_test)

print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_gnb))
print('\n')
print('Accuracy')
print(accuracy_score(y_test, y_pred_gnb))

#RandomForest

from sklearn.ensemble import RandomForestClassifier

ranfor = RandomForestClassifier(random_state=0,n_estimators=20,max_samples=200)
ranfor.fit(X_train, y_train)

y_pred_ranfor = ranfor.predict(X_test)

print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_ranfor))
print('\n')
print('Accuracy')
print(accuracy_score(y_test, y_pred_ranfor))

#SVM

from sklearn.svm import SVC
svm= SVC(probability=True)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_svm))
print('\n')
print('Accuracy')
print(accuracy_score(y_test, y_pred_svm))

# Voting Classifier without weights

from sklearn.ensemble import VotingClassifier

vc = VotingClassifier(estimators=[('DecisionTree',dectree),('KNN',knn),('GaussianNB',gnb),('RandomForest',ranfor),('SVM',svm)], 
                      voting='soft')
vc.fit(X_train, y_train)

y_pred_vc = vc.predict(X_test)

print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_vc))
print('\n')
print('Accuracy')
print(accuracy_score(y_test, y_pred_vc))

# Voting Classifier with weights

vc1 = VotingClassifier(estimators=[('DecisionTree',dectree),('KNN',knn),('GaussianNB',gnb),('RandomForest',ranfor),('SVM',svm)], 
                      voting='soft', weights=[1,2,2,2,1])
vc1.fit(X_train, y_train)

y_pred_vc1 = vc1.predict(X_test)

print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred_vc1))
print('\n')
print('Accuracy')
print(accuracy_score(y_test, y_pred_vc1))

print('Model Accuracy')
print('\n')
print('Decision Tree: '+str(round(accuracy_score(y_test, y_pred_dectree)*100,2))+'%')
print('GuassianNB: '+str(round(accuracy_score(y_test, y_pred_gnb)*100,2))+'%')
print('KNN: '+str(round(accuracy_score(y_test, y_pred_knn)*100,2))+'%')
print('\n')
print('Averaging Method')
print('Random Forest: '+str(round(accuracy_score(y_test, y_pred_ranfor)*100,2))+'%')
print('\n')
print('SVM Method')
print('SVC: '+str(round(accuracy_score(y_test, y_pred_svm)*100,2))+'%')
print('\n')
print('Voting Classifiers')
print('Voting Classifier without Weights: '+str(round(accuracy_score(y_test, y_pred_vc)*100,2))+'%')
print('Voting Classifier with Weights: '+str(round(accuracy_score(y_test, y_pred_vc1)*100,2))+'%')

# Creating a pickle file for the classifier
save_classifier = open("diabetes.pickle","wb")
pickle.dump(vc, save_classifier)
save_classifier.close()