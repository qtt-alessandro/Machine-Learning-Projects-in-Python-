import sys
import numpy as np 
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
from SVM_MultiClass_Q4 import SVM,Preprocessing_Pipeline
from sklearn.model_selection import train_test_split



def run_SVM_Q4():


    def accuracy(dataset,label_dataset,alphas_D,alphas_O,alphas_Q):

        smv_O_predictions = svm_O.multilabels_predict(dataset.values,alphas_O)
        smv_D_predictions = svm_D.multilabels_predict(dataset.values,alphas_D)
        smv_Q_predictions = svm_Q.multilabels_predict(dataset.values,alphas_Q)
        smv_ODQ_predictions = np.vstack((smv_O_predictions,smv_D_predictions,smv_Q_predictions )).T
        label_dataset["labels"] = " "
        label_dataset["labels"][label_dataset['letter'] == "O"] = 0
        label_dataset["labels"][label_dataset['letter'] == "D"] = 1
        label_dataset["labels"][label_dataset['letter'] == "Q"] = 2
        multiclass_pred = np.argmax(smv_ODQ_predictions,axis=1)
        accuracy = sum(multiclass_pred == label_dataset['labels'].values)/len(label_dataset['labels'].values)
        return multiclass_pred,accuracy

    pipeline = Preprocessing_Pipeline()
    data_qo,labels_qo = pipeline.load_dataset("Letters_Q_O.csv")
    data_d,labels_d = pipeline.load_dataset("Letter_D.csv")
    dataframes_dqo = [data_d,data_qo]
    lables_dqo = [labels_d,labels_qo]
    data_dqo   = pd.concat(dataframes_dqo)
    labels_dqo = pd.concat(lables_dqo)

    data_scl_dqo = pipeline.normal_scaling_dataset(data_dqo)
    data_scl_dqo_train, data_scl_dqo_test, labels_y_train, labels_y_test = train_test_split(data_scl_dqo,labels_dqo, test_size=0.75, random_state=1609286)

    labels_y_train = pipeline.label_encoding(labels_y_train,target="D")
    X_train, X_test, y_train, y_test = pipeline.splitting_dataset(data_scl_dqo_train,labels_y_train)
    svm_D = SVM(X_train, y_train, X_test, y_test, C=1, gamma=0.1)
    alphas_D = svm_D.fit()
    svm_D.predict(alphas_D)

    labels_y_train = pipeline.label_encoding(labels_y_train,target="Q")
    X_train, X_test, y_train, y_test = pipeline.splitting_dataset(data_scl_dqo_train,labels_y_train)
    svm_Q = SVM(X_train, y_train, X_test, y_test, C=1, gamma=0.1)
    alphas_Q = svm_Q.fit()
    svm_Q.predict(alphas_Q)

    labels_y_train = pipeline.label_encoding(labels_y_train,target="O")
    X_train, X_test, y_train, y_test = pipeline.splitting_dataset(data_scl_dqo_train,labels_y_train)
    svm_O = SVM(X_train, y_train, X_test, y_test, C=1, gamma=0.1)
    alphas_O = svm_O.fit()
    svm_O.predict(alphas_O)

    smv_ODQ_predictions_test,multiclass_test_accuracy = accuracy(data_scl_dqo_test,labels_y_test,alphas_D,alphas_O,alphas_Q)
    _,multiclass_train_accuracy = accuracy(data_scl_dqo_train,labels_y_train,alphas_D,alphas_O,alphas_Q)

    #####-------D
    print("================================================")
    print("=================LETTER D vs ALL================")
    print("================================================")
    print("C: ",svm_D.C)
    print("gamma value: ",svm_D.gamma)
    print("Number of iterations: ", svm_D.n_iters)
    print(f"difference between m(a) and M(a): ", "{:.6f}".format(svm_D.diff))
    print("Optimization Time: " + str(round(svm_D.elapsed_time,4)) + " sec")
    print("Train Accuracy Letter D vs all: ", "{:.6f}".format(svm_D.train_accuracy))
    print("Confusion Matrix - Train: \n", confusion_matrix(y_train, svm_D.train_y_pred))
    print("Test Accuracy Letter D vs all: ", "{:.6f}".format(svm_D.test_accuracy))
    print("Confusion Matrix: \n", confusion_matrix(y_test,svm_D.y_pred))
    print("================================================")
    #####-------

    #####-------Q
    print("=================LETTER Q vs ALL================")
    print("================================================")
    print("C: ",svm_Q.C)
    print("gamma value: ",svm_Q.gamma)
    print("Number of iterations: ", svm_Q.n_iters)
    print(f"difference between m(a) and M(a): ", "{:.6f}".format(svm_Q.diff))
    print("Optimization Time: " + str(round(svm_Q.elapsed_time,4)) + " sec")
    print("Train Accuracy Letter Q vs all: ", "{:.6f}".format(svm_Q.train_accuracy))
    print("Confusion Matrix - Train: \n", confusion_matrix(y_train, svm_Q.train_y_pred))
    print("Test Accuracy Letter Q vs all: ", "{:.6f}".format(svm_Q.test_accuracy))
    print("Confusion Matrix: \n", confusion_matrix(y_test,svm_Q.y_pred))
    print("================================================")
    #####-------

    #####-------O
    print("=================LETTER O vs ALL================")
    print("================================================")
    print("C: ",svm_O.C)
    print("gamma value: ",svm_O.gamma)
    print("Number of iterations: ", svm_O.n_iters)
    print(f"difference between m(a) and M(a): ", "{:.6f}".format(svm_O.diff))
    print("Optimization Time: " + str(round(svm_O.elapsed_time,4)) + " sec")
    print("Train Accuracy Letter O vs all: ", "{:.6f}".format(svm_O.train_accuracy))
    print("Confusion Matrix - Train: \n", confusion_matrix(y_train, svm_O.train_y_pred))
    print("Test Accuracy Letter O vs all: ", "{:.6f}".format(svm_O.test_accuracy))
    print("Confusion Matrix: \n", confusion_matrix(y_test,svm_O.y_pred))
    print("================================================")
    #####-------
    print("================MULTICLASS MODEL================")
    print("================================================")
    print("Multiclass Train Accuracy: ", multiclass_train_accuracy)

    print("Multiclass Test Accuracy: ", multiclass_test_accuracy)
    print("Total Elapsed Time: ",round(svm_D.elapsed_time+svm_O.elapsed_time+svm_Q.elapsed_time,4))
    c1 = [str(i) for i in labels_y_test['labels'].values]
    c2 = [str(i) for i in smv_ODQ_predictions_test]
    print("Confusion Matrix: \n", confusion_matrix(c1,c2,labels=["0","1","2"]))

if __name__ == "__main__":
    run_SVM_Q4()