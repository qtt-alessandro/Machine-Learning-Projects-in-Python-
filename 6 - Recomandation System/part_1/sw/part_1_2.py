import os
from surprise import Reader
from surprise import Dataset,KNNBaseline, SVD
from surprise.model_selection import KFold, cross_validate
from surprise.model_selection.search import GridSearchCV, RandomizedSearchCV

cwd = os.getcwd()

#------------------------DATASET_1_LOADING ---------------------------------------
file_path = os.path.expanduser('./Part_1/dataset/ratings_1.csv')
print("Loading Dataset...")
reader = Reader(line_format='user item rating', sep=',', rating_scale=[1, 5], skip_lines=1)
data_ratings_1 = Dataset.load_from_file(file_path, reader=reader)
print("Done.")
#----------------------------------------------------------------------------------

#------------------------DATASET_2_LOADING ----------------------------------------
file_path = os.path.expanduser('./Part_1/dataset/ratings_2.csv')
print("Loading Dataset...")
reader = Reader(line_format='user item rating', sep=',', rating_scale=[1, 10], skip_lines=1)
data_ratings_2 = Dataset.load_from_file(file_path, reader=reader)
print("Done.")
#-----------------------------------------------------------------------------------


data = [data_ratings_1, data_ratings_2]

#DATASET 1
#HYPER-PARAMTERS TUNING
search_params ={"k": [20,25,30,35,40,45,50],
                "min_k": [1,3,5],
                "sim_options": {
                    "name": ["cosine","pearson_baseline"],
                    "user_based":[True, False],
                    "min_support": [2,3,4]
                },
                "bsl_options":{
                    'method': ["sgd","als"],
                    'learning_rate': [0.001,0.005,0.01],
                    'n_epochs': [10,20,50],
                    'reg': [0.01,0.02,0.03],
                }

}

gs1 = RandomizedSearchCV(KNNBaseline, search_params, measures=['RMSE'], cv=5, n_jobs=4,joblib_verbose=1000)
gs1.fit(data[0])
#best score obtained
gs1.best_score
gs1.best_params


param_grid = {'n_factors': [98,100,102,104],
              'n_epochs': [10,20,50],
              'lr_all': [ 0.4, 0.01,0.5],
              'reg_all': [0.2,0.1,0.7,0.9]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=5, n_jobs=4,joblib_verbose=1000)
gs.fit(data[0])
gs.best_score
gs.best_params




#DATASET 2
#HYPER-PARAMTERS TUNING
search_params ={"k": [20,25,30,35,40,45,50],
                "min_k": [1,3,5],
                "sim_options": {
                    "name": ["cosine","pearson_baseline"],
                    "user_based":[True, False],
                    "min_support": [2,3,4]
                },
                "bsl_options":{
                    'method': ["sgd","als"],
                    'learning_rate': [0.001,0.005,0.01],
                    'n_epochs': [50,20],
                    'reg': [0.01,0.02,0.03],
                }

}

gs1 = RandomizedSearchCV(KNNBaseline, search_params, measures=['RMSE'], cv=5, n_jobs=4,joblib_verbose=1000)
gs1.fit(data[1])
#best score and parameters obtained
gs1.best_score
gs1.best_params






param_grid = {
              'n_factors': [98,100,102],
              'n_epochs': [10,20,50],
              'lr_all': [ 0.4, 0.01,0.5,0.05,0.001],
              'reg_all': [0.2,0.1,0.01,0.2]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=5, n_jobs=4,joblib_verbose=1000)
gs.fit(data[1])
gs.best_score
gs.best_params









