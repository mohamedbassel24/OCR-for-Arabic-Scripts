from sklearn.svm import SVC
from commonfunctions import *
from File_Accessing import *


def read_data(filename):
    data = []
    # TODO 1 : Read the file 'data1.csv' into the variable data.
    f = open(filename, "r")
    for line in f:
        data.append(line.split(','))

    for i in range(len(data)):
        for j in range(len(data[0])):
            data[i][j] = float(data[i][j])

    # print(data[0][0])
    # data contains the training data together with labelled classes.
    return data


def SVM_linear_training(X_train, y_train):  # X_train is the training features , y_train is the training labels , X_test is the test features
    svm_model_linear = SVC(kernel='linear', C=1).fit(X_train, y_train)
    #svm_predictions = svm_model_linear.predict(X_test)
    return svm_model_linear



def SVM_Non_linear_training(X_train, y_train):  # X_train is the training features , y_train is the training labels , X_test is the test features
    svclassifier = SVC(kernel='poly', degree=8).fit(X_train, y_train)
    # y_train=np.reshape(y_train,(len(y_train,1))
    #svclassifier.fit(X_train, y_train)
    #svm_predictions = svclassifier.predict(X_test)
    return svclassifier


def SVM_Gaussian(X_train, y_train):
    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(X_train, y_train)
    return svclassifier



def SVM_Classifier(SVM_Model,X_test):
    return SVM_Model.predict(X_test)






#X, Y = ReadModel("data_point")
#print(X)
#print(Y)
#X1=np.matrix([[1, 2,3], [3, 4,6]])
#y1=[0,1]
#X2=np.matrix([[7,4,1],[3,4,6]])
#model1=SVM_linear_training(X1,y1)
#model2=SVM_Non_linear_training(X1,y1)
#y2=SVM_Classifier(model1,X2)
#y3=SVM_Classifier(model2,X2)
#print(y2)
#print(y3)
