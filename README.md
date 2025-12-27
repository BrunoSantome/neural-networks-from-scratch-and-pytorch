# Programming_math_Ai_Assesmment_Msc_Ai
Repository to develop all the tasks within the final assessment of the module "Porgramming and Math for AI" in the Msc Artificial Inteligence. 

Important: recreate the results:  

To recreate the results it is mandatory to change the PATH variable that points to the two different datasets used for both tasks. This PATH variable is located in both python scripts called preprocessing.py which are in charge of loading the data and make the correct transformation for the posterior use in the models. 

A seed has been set for all results, value of the seed = 42. 

Structure of the project: 

- Dataset folder
There is a folder containing the datasets used for both Task 1 and Task 2, namely Stellar_Object_Classification and Aerial_Landscapes, respectively. The other two datasets were also used during Task 1, but since they achieved very high performance without much tuning, they were ultimately discarded. 

- Images & Images_2

Both folders contain plots and images generated during the experiments on task 1 and task 2 respecively.

- Src folder

This folder is where all the development of the assessment has been carried out, there is two sub-folders: dev_task1 and dev_task2. 

EXCEL file hyperparameter_tuning_results.xlsx

This file has been used to store all the different variation of hyperparameters of the task 1 together with results (loss, train accuracy, test accuracy) and the amount of processing time that each test has taken. It is not for the results of task 2. 

- dev_task1 sub-folder 

Folder with all the python scripts used to develop and test the neural network with numpy.

preprocessing.py: python script to load the data, pre-process it and split it in train and test sets, ready for training a model. 

dnn_functions.py: python script where all the activation functions, loss functions and some plot functions have been implemented, these are used and imported by the neural network class. 

deep_nn.py: python script with a customizable Neural network class. This class implements the main part of the assessment; parameters initialisation, forward pass, loss calculations, backward pass, parameters update and model evaluation. Everything is commented and well explain on the script. 

dnn_tests.py: python script that has been used as the main playground for unit tests during the development of the neural network and hyperparameter tuning during once all the parts were correctly implemented and joint. 

- dev_task2 sub-folder

preprocessing.py: python script that is in charge of loading the data, splitting it between train and test set, transform both sets and pass them to a Dataloader. Make sure the data is ready and efficiently passed for using in models. 

pytorch_nn_dev.py: python script that implements the whole of task 2, starting by the neural network class model following the Convolutional neural network class and set of functions to tune the models. There is a long set of comments which has been the different outcomes of several trials with both models in order to find the final solution.


