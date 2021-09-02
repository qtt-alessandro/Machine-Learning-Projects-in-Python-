import os
from surprise import Reader
from surprise import Dataset, NormalPredictor, BaselineOnly, KNNBasic, KNNWithMeans, KNNBaseline, KNNWithZScore, SVD, SVDpp, CoClustering, SlopeOne, NMF
from surprise.model_selection import KFold, cross_validate
from surprise.model_selection.search import GridSearchCV, RandomizedSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cwd = os.getcwd()

def select_algorithm():

    val = input("Select the dataset to load: \n 1: Ratings Dataset 1 \n 2: Ratings Dataset 2")
    val = int(val)
    if val == 1:
        # ------------------------DATASET_1_LOADING ---------------------------------------
        file_path = os.path.expanduser('./Part_1/dataset/ratings_1.csv')
        print("Loading Dataset...")
        reader = Reader(line_format='user item rating', sep=',', rating_scale=[1, 5], skip_lines=1)
        data = Dataset.load_from_file(file_path, reader=reader)
        print("Done.")
    # ----------------------------------------------------------------------------------
    if val == 2:
        # ------------------------DATASET_2_LOADING ---------------------------------------
        file_path = os.path.expanduser('./Part_1/dataset/ratings_2.csv')
        print("Loading Dataset...")
        reader = Reader(line_format='user item rating', sep=',', rating_scale=[1, 10], skip_lines=1)
        data = Dataset.load_from_file(file_path, reader=reader)
        print("Done.")
        # ----------------------------------------------------------------------------------

    alg = input(" Select the algorithm: \n 1 : NORMAL PREDICTOR \n 2 : BASELINE_PREDICTION \n 3 : COCLUSTERING \n 4 : SLOPE ONE \n 5 : NMF \n 6 : KNNBASIC + KNNWITHMEANS + KNNWITHZSCORE \n 7 : KNNBASELINE \n 8 : SVD \n 9 : SVD++ \n")
    alg = int(alg)
    print("computing...")

    if alg == 1:
        #-------------------1.NORMAL PREDICTOR---------------------------------------------
        current_algo = NormalPredictor()
        kf = KFold(n_splits=5, random_state=123)
        g = cross_validate(current_algo, data, measures=['RMSE'], cv=kf,n_jobs=4,verbose=True)
        #-----------------------------------------------------------------------------------

    if alg == 2:
        #-------------------2.BASELINE_PREDICTION-------------------------------------------
        baseline_predictor_options = {
        'method': "sgd",
        'learning_rate': 0.005,
        'n_epochs': 50,
        'reg': 0.02,
        }
        current_algo = BaselineOnly(bsl_options=baseline_predictor_options, verbose=True)
        kf = KFold(n_splits=5, random_state=123)
        cross_validate(current_algo, data, measures=['RMSE'], cv=kf,n_jobs=4,verbose=True)
        #-------------------------------------------------------------------------------------

    if alg == 3:
        #----------------------3.COCLUSTERING-------------------------------------------------
        current_algo = CoClustering(n_cltr_u=3, n_cltr_i =3,n_epochs = 20, random_state = 123)
        kf = KFold(n_splits=5, random_state=123)
        g = cross_validate(current_algo, data, measures=['RMSE'], cv=kf,n_jobs=4,verbose=True)
        #--------------------------------------------------------------------------------------

    if alg == 4:
        #-----------------------4.SLOPE ONE----------------------------------------------------
        current_algo = SlopeOne()
        kf = KFold(n_splits=5, random_state=123)
        g = cross_validate(current_algo, data, measures=['RMSE'], cv=kf,n_jobs=4,verbose=True)
        #----------------------------------------------------------------------------------------

    if alg == 5:
        #-------------------------5.NMF----------------------------------------------------------
        current_algo = NMF()
        kf = KFold(n_splits=5, random_state=123)
        g = cross_validate(current_algo, data, measures=['RMSE'], cv=kf,n_jobs=4,verbose=True)
        #----------------------------------------------------------------------------------------

    if alg == 6:
        #---------------------6.KNNBASIC,7.KNNWITHMEANS,8.KNNWITHZSCORE-------------------------------
        MAXIMUM_number_of_neighbors_to_consider = 40
        min_number_of_neighbors_to_consider = 1
        similarity_options = {'name': "cosine",
                              'user_based': False,
                              'min_support': 3,
                              }
        kf = KFold(n_splits=5, random_state=123)
        for current_algo in (KNNBasic, KNNWithMeans, KNNWithZScore):
            current_algo = current_algo(k=MAXIMUM_number_of_neighbors_to_consider, min_k=min_number_of_neighbors_to_consider,
                        sim_options=similarity_options, verbose=True)
            cross_validate(current_algo, data, measures=['RMSE'], cv=kf,n_jobs=4, verbose=True)
        #------------------------------------------------------------------------------------------------

    if alg == 7:
        #----------------------9.KNNBASELINE--------------------------------------------------------------
        MAXIMUM_number_of_neighbors_to_consider = 40
        min_number_of_neighbors_to_consider = 1
        similarity_options = {
            'name': "cosine",
            'user_based': False,
            'min_support': 3,
        }

        baseline_predictor_options = {
            'method': "sgd",
            'learning_rate': 0.005,
            'n_epochs': 50,
            'reg': 0.02,
        }

        current_algo = KNNBaseline(k=MAXIMUM_number_of_neighbors_to_consider, min_k=min_number_of_neighbors_to_consider,
                           sim_options=similarity_options, bls_options=baseline_predictor_options, verbose=True)
        kf = KFold(n_splits=5, random_state=123)
        cross_validate(current_algo, data, measures=['RMSE'], cv=kf,n_jobs=4, verbose=True)
        #----------------------------------------------------------------------------------------

    if alg == 8:
        #--------------------------10.SVD--------------------------------------------------------
        current_algo = SVD(n_factors = 100, n_epochs=50,lr_all=0.005,reg_all=0.02, verbose=True)
        kf = KFold(n_splits=5, random_state=123)
        cross_validate(current_algo, data, measures=['RMSE'], cv=kf,n_jobs=4, verbose=True)
        #-----------------------------------------------------------------------------------------


    if alg == 9:
        #---------------------------11.SVD++------------------------------------------------------
        current_algo = SVDpp(n_factors = 100, n_epochs=20,lr_all=0.005,reg_all=0.02, verbose=True)
        kf = KFold(n_splits=5, random_state=123)
        cross_validate(current_algo, data, measures=['RMSE'], cv=kf,n_jobs=4, verbose=True)
        #-----------------------------------------------------------------------------------------


#---------------------RUN THE FUNCTION AND SELECT THE ALGORITHMS----------------------------------
select_algorithm()
#---------------------RUN THE FUNCTION AND SELECT THE ALGORITHMS----------------------------------


#Dataset 1
x_1  = np.array([1, 2, 3, 4, 5, 6, 7,8,9,10,11])
y_1  = np.array([1.5134, 0.9163,0.9406,0.9233,0.9365, 1.0069, 0.9169, 0.9194, 0.9148, 0.9071,0.8987])
sd_1 = np.array([0.0054,0.0039,0.0088,0.0031,0.0043,0.0047,0.0034,0.0035,0.0044,0.0030,0.0033])
diff_1 = y_1-sd_1
my_xticks = ["Normal Predictor", "BaselineOnly","CoClustering"," Slope One","NMF","KNNBasic", "KNN Means", "KNN Zscore"
             ,"KNNBaseLine", "SVD","SVD++"]
plt.xticks(x_1, my_xticks, rotation = 45)
plt.scatter(x_1, diff_1)
plt.title("Algorithms Performances evaluated with RMSE Dataset 1")
plt.ylabel('RMSE')
plt.savefig('algo_performances_1.png')
plt.show()


d_1 = {'Algorithms': my_xticks,
     'RMSE': y_1.tolist(),
     'std': sd_1.tolist(),
     'RMSE-std': diff_1.tolist()
     }
df_1 = pd.DataFrame(data=d_1)
df_1.sort_values(by=['RMSE-std'])

#Dataset 2
x_2  = np.array([1, 2, 3, 4, 5, 6, 7,8,9,10,11])
y_2  = np.array([3.2211,1.9845,2.0347,1.9823,1.9927,2.3845,1.9781,1.9868,1.9898,1.9002,1.8271])
sd_2 = np.array([0.0255,0.0096,0.00179,0.0096,0.0264,0.0221,0.0130,0.0122,0.0126,0.0230,0.0170])
diff_2 = y_2-sd_2
my_xticks = ["Normal Predictor", "BaselineOnly","CoClustering"," Slope One","NMF","KNNBasic", "KNN Means", "KNN Zscore"
             ,"KNNBaseLine", "SVD","SVD++"]
plt.xticks(x_2, my_xticks, rotation = 45)
plt.scatter(x_2, diff_2)
plt.title("Algorithms Performances evaluated with RMSE Dataset 2")
plt.ylabel('RMSE')
plt.savefig('algo_performances_2.png')
plt.show()

d_2 = {'Algorithms': my_xticks,
     'RMSE': y_2.tolist(),
     'std': sd_2.tolist(),
     'RMSE-std': diff_2.tolist()
     }
df_2 = pd.DataFrame(data=d_2)
df_2.sort_values(by=['RMSE-std'])
