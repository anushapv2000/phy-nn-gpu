import uproot3
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# extracting root files

train_file = uproot3.open('train_D02kpipi0vxVc-cont0p5.root')
train_tree = train_file['d0tree;1']
train_branches = train_tree.arrays(namedecode = 'utf-8')
#train_ranches.keys()

test_file = uproot3.open('test_D02kpipi0vxVc-cont0p5.root')
test_tree = test_file['d0tree;1']
test_branches = test_tree.arrays(namedecode = 'utf-8')
#test_branches.keys()

# storing root file's content into pandas variable

train_dataset = train_tree.pandas.df(["flightTime","vcosxy","vcosHP","vchiProb","vangAZ","K_kaonID","pi_pionID","pi0_ocosHL","pi0_chiProb","pi0_maxCT","M_Kpi","M_Kpi0","isSignal"])
test_dataset = test_tree.pandas.df(["flightTime","vcosxy","vcosHP","vchiProb","vangAZ","K_kaonID","pi_pionID","pi0_ocosHL","pi0_chiProb","pi0_maxCT","M_Kpi","M_Kpi0","isSignal"])

#importing the Dataset to variables

X_train = train_dataset.iloc[:,:-1].values
y_train = train_dataset.iloc[:,-1].values

X_test = test_dataset.iloc[:,:-1].values
y_test = test_dataset.iloc[:,-1].values

#Feature Scalling 

# Scalling Data ofr neural network

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Neural Netwoork
import timeit
starttime = timeit.default_timer()
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units =13, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units =10, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units =10, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units =9, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units =1, activation = 'sigmoid'))
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann.fit(X_train, y_train, batch_size=32, epochs=100)
print("time taken to train the model: ",timeit.default_timer()-starttime," seconds\n\n")
print(ann.summary())
# predicting Test Results

y_pred = ann.predict(X_test)
y_pred1 = (y_pred > 0.5)

#Storing the predict values in csv file for further use
df = pd.DataFrame(y_pred)
np.save('nn1.npy',y_pred)
print(df)

#Macking the confusion Matrix and finding the accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred1)
print("Confusion Matrix:\n")
print(cm)
acc_score = accuracy_score(y_pred1, y_test)


#PLoting the ROC curve

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics 

#function to plot the ROC curve

def plot_roc_curve(fpr, tpr, area):
	plt.plot(fpr, tpr, color='green', label='Neural Network - area under curve: %0.2f' % area)
	plt.plot([0,1], [0,1], color='darkblue', linestyle='--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('true Positve Rate')
	plt.title('Receiver Operating Characteristics (ROC) Curve')
	plt.legend()
	plt.show()
	plt.savefig('ROC curve2.pdf')
	plt.close()


fpr, tpr, threshold = roc_curve(y_test, y_pred)
area = metrics.auc(fpr, tpr)
plot_roc_curve(fpr,tpr,area)
