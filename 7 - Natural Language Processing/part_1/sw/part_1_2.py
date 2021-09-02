import json
import os
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from pandas import DataFrame
import pandas as pd
import time
import pickle
from scipy.stats import loguniform
from sklearn.linear_model import LogisticRegression

def read_pickle(name):
    with open( str(name)+'.pickle', 'rb') as handle:
        data = pickle.load(handle)
        return data

# Storing pickle files as dataframes:
train = read_pickle("./train_final")
test = read_pickle("./test_final")
dev = read_pickle("./dev_final")

# According to the output, refutes and supports, output column is updated as 0 and 1 accordingly
train.output = train.output.apply(str)
train['output'] = train['output'].replace(str(["{'answer': 'SUPPORTS'}"]), 1)
train['output'] = train['output'].replace(str(["{'answer': 'REFUTES'}"]), 0)
dev.output = dev.output.apply(str)
dev['output'] = dev['output'].replace(str(["{'answer': 'SUPPORTS'}"]), 1)
dev['output'] = dev['output'].replace(str(["{'answer': 'REFUTES'}"]), 0)

# The next step is taken because the same dataframes will be used for the 2nd part
# so, we will eliminate the nan values appearing in the abstract_embedding column
train = train.dropna()
train = train.reset_index()

# Now, label frequency is aimed to be observed in order to have balanced dataset:
print(len(train[train['output'] == 1])/len(train), '% data labeled as 1.')
print(len(train[train['output'] == 0])/len(train), '% data labeled as 0.')

# Then, we will take the portion of train dataset that balance data
to_remove = np.random.choice(train[train['output']==1].index,
                             size=(len(train[train['output'] == 1])-len(train[train['output'] == 0])),replace=False)
train = train.drop(to_remove)

# It can be observed that we have same number of data points for each label
print(len(train[train['output'] == 1])/len(train), '% data labeled as 1.')
print(len(train[train['output'] == 0])/len(train), '% data labeled as 0.')
train = train.reset_index()

# We take the next step to use columns as variable in the following algorithms
train[np.arange(0,len(train['claim_embedding'][0]))] = pd.DataFrame(train.claim_embedding.tolist())
test[np.arange(0,len(test['claim_embedding'][0]))] = pd.DataFrame(test.claim_embedding.tolist())
dev[np.arange(0,len(dev['claim_embedding'][0]))] = pd.DataFrame(dev.claim_embedding.tolist())

X_train = train[list(range(0, 768))]
Y_train = DataFrame(train, columns = ['output'])
X_dev = dev[list(range(0, 768))]
Y_dev = DataFrame(dev, columns = ['output'])
X_test = test[list(range(0, 768))]

# It's time to apply the classifiers

# Firstly, We will try logistic regression
start = time.time()

param_log = {'solver': ['newton-cg', 'lbfgs', 'liblinear'],
              'penalty': ['l1', 'l2', 'elasticnet'],
              'C': loguniform(1e-5, 100)}
grid_log = RandomizedSearchCV(LogisticRegression(), param_log, n_iter=500, n_jobs=4, cv=5, random_state=1,verbose=100)
grid_log.fit(X_train, Y_train)
grid_best_param = grid_log.best_params_

# test the best parameters in the validation dataset
y_pred_log = grid_log.predict(X_dev)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(Y_dev, y_pred_log))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(Y_dev, y_pred_log))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(Y_dev, y_pred_log))

end = time.time()
print((end - start)/60)

# Confusion Matrix
print(confusion_matrix(Y_dev, y_pred_log))

# Best parameters
print(grid_best_param)

# Testing for the test set:
test_y_pred_log = grid_log.predict(X_test)

# writing to jsonl
id_test_log = []
for i in tqdm(range(len(test))):
    answer = ''
    if test_y_pred_log[i] == 1:
        answer = "SUPPORTS"
    else:
        answer = "REFUTES"
    id_test_log.append({"id": str(test['id'][i]), 'output': [{"answer": answer}]})

with open('./test_set_pred_1.jsonl', 'w') as outfile:
    for entry in tqdm(id_test_log):
        json.dump(entry, outfile)
        outfile.write('\n')


# Secondly, We will continue with knn
start = time.time()

grid_params = {'n_neighbors' : [13,15,17],
               'weights' : ['uniform', 'distance'],
               'metric' : ['euclidean', 'manhattan']}
gs = RandomizedSearchCV(KNeighborsClassifier(), grid_params, verbose=3, cv=5, n_jobs=4)
gs.fit(X_train, Y_train)
best_param_grid = gs.best_params_

y_pred = gs.predict(X_dev)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(Y_dev, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(Y_dev, y_pred))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(Y_dev, y_pred))

end = time.time()
print((end - start)/60)

# Confusion Matrix
confusion_matrix(Y_dev, y_pred)
# Best parameters
print(best_param_grid)

# Testing for the test set:
knn_y_pred = gs.predict(X_test)

# writing to jsonl
id_test = []
for i in tqdm(range(len(test))):
    answer = ''
    if knn_y_pred[i]==1:
        answer = "SUPPORTS"
    else:
        answer = "REFUTES"
    id_test.append({"id": str(test['id'][i]), 'output': [{"answer": answer}]})

with open('./input_data/test_set_pred_2.jsonl', 'w') as outfile:
    for entry in tqdm(id_test):
        json.dump(entry, outfile)
        outfile.write('\n')
