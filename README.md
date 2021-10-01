# Conda Environment for Neural Diabetes

>About Conda Environments: [https://conda.io/docs/user-guide/tasks/manage-environments.html </br>
>Installing git: [https://git-scm.com/book/en/v2/Getting-Started-Installing-Git]

***Quick Start :Setup the Environment***

Note that for the code above to work,you need to be in the directory where the environment.yml file lives so 'cd' to that directory first

The environment name is neural-pkg as defined in the environment.yml file.</br>
To begin, install the environment using:</br>
> ```conda env create -f environment.yml```

Activate the environment</br>
>  ````conda activate package````

 update the environment at any time using:</br>
 >````conda env update -f environment.yml````

To view the environments installed</br>
>  ````conda info --envs````
  



# About the Program:

This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to predict based on diagnostic measurements whether a patient has diabetes.



INPUT VARIABLES(X):</br>
1.Pregnancies: Number of times pregnant</br>
2.Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test</br>
3.BloodPressure: Diastolic blood pressure (mm Hg)</br>
4.SkinThickness: Triceps skin fold thickness (mm)</br>
5.Insulin: 2-Hour serum insulin (mu U/ml)</br>
6.BMI: Body mass index (weight in kg/(height in m)^2)</br>
7.DiabetesPedigreeFunction: Diabetes pedigree function</br>
8.Age: Age (years)</br>

OUPTUT VARIABLE(y):</br>
Outcome: Class variable (0 or 1)

Steps:</br>
>-Load Data</br>
-Keras Model is used as a sequence of layers</br>
-The model is fed with rows of data with 8 variables</br>
-The first hidden layer has 12 nodes and relu activation function is used</br>
-Second Hidden layer consist of 8 nodes and relu activation is used</br>
-The output layer has one node and use sigmoid function.</br>

Compiled and evaluated</br>
>Binary Cross-Entropy Loss:</br>
The default loss function used for binary classification problems.Cross-entropy will calculate a score that summarizes the average difference between the actual and predicted probability distributions for predicting class 1. The score is minimized and a perfect cross-entropy value is 0

>Optimizer:</br>
The choice of optimixation algorithm for the deep learning model can mean the difference between good results .Adam optimizer can be used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data.

>Accuracy is claculated using metrics argument.</br>

>Keras Model is used on the loaded data by using fit() function.Here data training over epochs and epochs is split into batches.After the training ,model is evaluated using evaluate() function which return the loss of the model and accuracy of the model.

Make Predictions</br>
>As Sigmoid activation functions are used output value will be in the range of 0 and 1,hence it is rounded off .Prediction is done by calling the predict() function on the model</br>
Running the code shows a message for 150 epochs printing the loss and accuracy.</br>
The result provide the actual and predicted values.
