# phy-nn-gpu
This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to predict based on diagnostic measurements whether a patient has diabetes.


INPUT VARIABLES(X):
Pregnancies: Number of times pregnant
Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
BloodPressure: Diastolic blood pressure (mm Hg)
SkinThickness: Triceps skin fold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction: Diabetes pedigree function
Age: Age (years)
OUPTUT VARIABLE(y):
Outcome: Class variable (0 or 1)

Steps:
Load Data
Keras Model is used as a sequence of layers
-The model is fed with rows of data with 8 variables
-The first hidden layer has 12 nodes and relu activation function is used
-Second Hidden layer consist of 8 nodes and relu activation is used
-The output layer has one node and use sigmoid function.

Compiled and evaluated
Binary Cross-Entropy Loss:
The default loss function used for binary classification problems.Cross-entropy will calculate a score that summarizes the average difference between the actual and predicted probability distributions for predicting class 1. The score is minimized and a perfect cross-entropy value is 0

Optimizer:
The choice of optimixation algorithm for the deep learning model can mean the difference between good results .Adam optimizer can be used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data.

Accuracy is claculated using metrics argument.
Keras Model is used on the loaded data by using fit() function.Here data training over epochs and epochs is split into batches.

Make Predictions.
As Sigmoid activation functions are used output value will be in the range of 0 and 1,hence it is rounded off 
