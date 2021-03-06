#Neural Network project
import pandas as pd
import numpy as np
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from os import getcwd

workpath=getcwd()

dataset = loadtxt(workpath+'/data/pima-indians-diabetes.csv', delimiter=',')
print(dataset.shape)
x=dataset[:,0:8]
y=dataset[:,8]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=150,batch_size=50)
_,accuracy=model.evaluate(x_test,y_test,verbose=0)
#print('Accuracy%.2f'%(accuracy*100))
print(accuracy)

pred=model.predict(x_test)
y_pred=np.round(pred).astype(int)
print(pred.dtype)
df=pd.DataFrame({'Actual':y_test.flatten(),
                 'Predicted':y_pred.flatten()})
print(df)
