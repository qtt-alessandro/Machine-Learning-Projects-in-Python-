import sys,os
from SVM_Q1 import SVM,Preprocessing_Pipeline
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from tqdm import tqdm
import pandas as pd 
import numpy as np 
import operator 
import itertools
import pickle
import cvxopt
import time

class Jobs_Parallelizer(object):
        
    @staticmethod
    def cartesian_combination(dict):
        """
        function to implement the cartesian product considering all the 
        possibilities for C and gamma
        :input -> dictionary
        :output -> list 
        """
        
        C = dict["C"]
        gamma= dict["gamma"] 

        combination = [C,gamma]
        
        

        cartesian_stack = []
        for element in itertools.product(*combination):
            cartesian_stack.append(element)
        return cartesian_stack
    
    @staticmethod
    def hyper_params_tune(X,Y,list,num_fold,n_epochs,n_jobs=-1): 
        """
        this function compute the hyperparameters tuning according to 
        the cartesian product of gamma and C
        :input -> X,Y, number of fold for the cross validation, epochs and cpu cores for the workload splitting
        """
        
        def cross_validation(i):
            """
            sub-fuction that thakes the index of the iteration to parallelize on multiple cores
            :output -> flatten list with the results of the hyper-parameters tuning
            """
            C_it = list[i][0]
            gamma_it = list[i][1]
            KFold_accuracy_epoch = []
            KFold_accuracy = []

            for i in range(n_epochs):
                kf = KFold(n_splits=num_fold, shuffle=True)
                for train_index, val_index in kf.split(X,Y):

                    X_train, X_test = X[train_index], X[val_index]
                    y_train, y_test = Y[train_index], Y[val_index]


                    svm = SVM(X_train, y_train, X_test, y_test, C=C_it, gamma=gamma_it)
                    alphas = svm.fit()
                    svm.predict(alphas)

                    KFold_accuracy_epoch.append(svm.test_accuracy)

                KFold_accuracy.append(sum(KFold_accuracy_epoch)/len(KFold_accuracy_epoch))

            return sum(KFold_accuracy)/len(KFold_accuracy)
    
        flatten_list = Parallel(n_jobs=n_jobs)(delayed(cross_validation)(i) for i in tqdm(range(len(list))))
        return flatten_list
    
    @staticmethod
    def write_pickle(name,list):
        """
        save the result list from the previous function as pickle file 
        """
        with open(name, 'wb') as f: 
            pickle.dump(list, f)


    @staticmethod
    def read_pickle(name):
        """
        read the result list of the hyper-parameters tuning from pickle file
        """
        with open(name, 'rb') as f:
            return pickle.load(f)


if __name__ == "__main__":
    #calling the preprocessing pipeline that enable the normal scaling of the data and the encoding of the output features
    pipeline = Preprocessing_Pipeline()
    data,labels = pipeline.load_dataset("Letters_Q_O.csv")
    data_scl = pipeline.normal_scaling_dataset(data)
    label_enc = pipeline.label_encoding(labels,target="Q")
    X_train, X_test, y_train, y_test = pipeline.splitting_dataset(data_scl,label_enc)


    params = {"C": [1],
            "gamma": [0.001,0.005,0.0085,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.8,1,2]}

    #parallelizing the workload calling Job_Parallelizer class that enable the splitting over multiples cores.
    
    params_list = Jobs_Parallelizer.cartesian_combination(params)
    print("Total Number of iterations: ", len(params_list))
    HP_result_dict = {}
    start = time.time()
    #list with the results from the hyper parameters tuning obatained from 5 K-Fold crossvalidation and 10 epochs
    #  to denoise the 
    result = Jobs_Parallelizer.hyper_params_tune(X_train,y_train,params_list,num_fold=5,n_epochs=10,n_jobs=-1)
    #building and updating dictionary with a tuple as key and the accuracy as value.
    HP_result_dict = HP_result_dict.fromkeys(params_list, 0)
    HP_result_dict.update(zip(HP_result_dict, result))
    HP_result_frame = pd.DataFrame(HP_result_dict.items(),columns=['hyper_params','val_acc'])
    HP_result_frame=HP_result_frame.sort_values(by='val_acc',ascending=False)
    Jobs_Parallelizer.write_pickle('HP_sorted_list_Q1_QQ.pkl',HP_result_frame)
    end = time.time()
    print('Elapsed Time: ', (end - start)/60)