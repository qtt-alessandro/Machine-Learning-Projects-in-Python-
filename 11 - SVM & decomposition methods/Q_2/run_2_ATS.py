import sys
import numpy as np 
from sklearn.metrics import confusion_matrix
from SVM_DM_Q2 import SVM_DM,Preprocessing_Pipeline

def run_SVM_Q2():
    pipeline = Preprocessing_Pipeline()
    data,labels = pipeline.load_dataset("Letters_Q_O.csv")
    data_scl = pipeline.normal_scaling_dataset(data)
    label_enc = pipeline.label_encoding(labels,target="Q")
    X_train, X_test, y_train, y_test = pipeline.splitting_dataset(data_scl,label_enc)
    svm = SVM_DM(X_train,y_train,X_test,y_test,C=1,gamma=0.1)
    alphas = svm.fit()
    svm.predict(alphas)
    print("C: ",svm.C)
    print("gamma value: ",svm.gamma)
    print("Number of iterations: ", svm.n_iters)
    print("Number of iterations (CVXOPT): ", svm.n_iters_solver)
    print("Optimization Time: " + str(round(svm.elapsed_time, 4)) + " sec")
    print(f"difference between m(a) and M(a): ", "{:.6f}".format(svm.diff))
    print(f"value of q: {svm.qq}")

    print("Train Accuracy: ", "{:.6f}".format(svm.train_accuracy))
    print("Confusion Matrix - Train: \n", confusion_matrix(y_train,svm.train_y_pred))
    
    print("Test Accuracy: ", "{:.6f}".format(svm.test_accuracy))
    print("Confusion Matrix - Test: \n", confusion_matrix(y_test,svm.y_pred))

if __name__ == "__main__":
    run_SVM_Q2()
