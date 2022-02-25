import numpy as np 
import cvxopt
import time 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Preprocessing_Pipeline(object):
    
    def __init__(self):
        
        self.x_cols = ['x-box', 'y-box', 'width', 'high', 'onpix', 'x-bar', 'y-bar', 'x2bar',
                    'y2bar', 'xybar', 'x2ybr', 'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx']
        
        self.y_cols = ['letter']
        
    def load_dataset(self,name):
        #loading the datasets
        X = pd.read_csv('./Data/'+ str(name),usecols=self.x_cols)
        Y = pd.read_csv('./Data/'+ str(name),usecols=self.y_cols)
        return X,Y
    
    
    def label_encoding(self,Y,target):
        #encoding the labels of the features
        Y["labels"] = np.where(Y['letter'].str.contains(str(target)),1,-1)
        return Y 
        

    def normal_scaling_dataset(self,X):
        """
        using normal scaling from sklearn to normalize the input features, as required for 
        a SVM classification problem
        """
        scaler=StandardScaler()
        X_scaled=scaler.fit_transform(X)
        X_scaled_dataframe = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        return X_scaled_dataframe
    
    def splitting_dataset(self,X,Y,test_size=0.2,seed=1609286):

        #splitting the dataset according to the seed 
        X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=test_size, random_state=seed)

        return X_train.values, X_test.values, y_train['labels'].values, y_test['labels'].values

class SVM(object):
    def __init__(self, x, y, x_test, y_test, C, gamma):
        # data
        self.x = x
        self.y = y 
        self.P, _ = self.x.shape
        self.x_test = x_test
        self.y_test = y_test
        # hyper parameters
        self.C = C
        self.gamma = gamma
        self.epsilon = 1e-4


    @staticmethod
    def Kernel(x1, x2, gamma):
        '''
        Gaussian Kernel.
        :return G.K. matrix -> (x1_dim_0, x2_dim_0)
        '''
        diff = np.expand_dims(x1,axis=2) - x2.T # (Px1, n_features, Px2)
        return np.exp(-gamma * np.sum(diff**2, axis=1)) # (Px1, Px2)

    def _return_RS(self):
        """
        Return the R and S sets that will help decomposition methods to choose violating pairs.
        :param alpha: vector of length P
        :return: R and S sets(numpy vectors) that are candidates for the Decomposition method's violating pairs
        """
        alpha = self.alpha
        a_less_C = alpha < self.C - self.epsilon
        a_greater_zero = alpha > 0 + self.epsilon
        y_equals_1 = self.y == 1
        y_equals_1 = y_equals_1.reshape(a_greater_zero.shape)
        y_equals_minus_1 = self.y == -1
        y_equals_minus_1 = y_equals_minus_1.reshape(a_greater_zero.shape)
        R_mask = (a_less_C & y_equals_1) | (a_greater_zero & y_equals_minus_1)
        R = np.array([i for i, x in enumerate(R_mask) if x])
        S_mask = (a_less_C & y_equals_minus_1) | (a_greater_zero & y_equals_1)
        S = np.array([i for i, x in enumerate(S_mask) if x])

        return R, S

    def _get_gradient(self, Q, alpha, old_alpha):
        """
        :return the updated gradient of the k-th iteration
        """
        out = -1 * np.ones(self.P) + ( Q @ (alpha - old_alpha) )
        return out

    def _get_minmax(self, R, S, grad):

        """
        :return subsets m,M
        """
        m = np.max(-1 * grad[R] * self.y[R])
        M = np.min(-1 * grad[S] * self.y[S])
        return m, M
    
    def fit(self,show_progress=False):

        """
        The fit function enable the resolution of SVM classification problem using CVXOPT, 
        using the matrices defined as following and as highlighted in the report.

                                            1 
                                        min - a^T H a  - a
                                         a  2

                                         s.t -a_i <= 0
                                              a_i < C 
                                              y^T a = 0 
        """

        grad = -1 * np.ones(self.P)
        alpha_zero = np.zeros(self.P)
        
        #creating the RBF Kernel matrix 
        self.K = SVM.Kernel(self.x,self.x,self.gamma) 

        #-----
        #shape: 1068x1068 -> (P,P)
        Q = cvxopt.matrix(np.outer(self.y, self.y) * self.K)
        #-----

        #-----
        #shape: (1068,1) -> (P,1)
        q_tmp = -np.ones((self.P,1))
        q = cvxopt.matrix(q_tmp)
        #-----
        
        #-----
        #shape: 1068x1068 -> (P,P)
        block_1_G = -np.eye(self.P)
        #shape: 1068x1068 -> (P,P)
        block_2_G = np.eye(self.P)
        #-----

        #-----
        #shape: 2136x1068 -> (2xP,P)
        G_tmp = np.concatenate((block_1_G,block_2_G), axis=0)
        G = cvxopt.matrix(G_tmp)
        #-----

        #-----
        #shape: (1068,1) -> (P,1)
        block_1_h = np.zeros((self.P))
        #shape: (1068,1) -> (P,1)
        block_2_h = self.C*np.ones((self.P))
        h_tmp = np.concatenate((block_1_h,block_2_h),axis= 0)
        #shape: (2136,1) -> (2xP,1)
        h = cvxopt.matrix(h_tmp)
        #-----

        #-----
        #shape: (1,1068) -> (1,P)
        A = cvxopt.matrix(self.y, (1,self.P), "d")
        #-----

        #-----
        #shape: (1,1) -> (1,1)
        b = cvxopt.matrix(np.zeros(1))
        #-----
        
        ## CVOPT Solver -- optional paramters
        if show_progress:
            cvxopt.solvers.options["show_progress"] = True
        else:
            cvxopt.solvers.options["show_progress"] = False
            
        start = time.time()
        #passing the matrices that define the quadratic problem into the optimizer
        sol = cvxopt.solvers.qp(Q, q, G, h, A, b)
        end = time.time()
        self.elapsed_time= end-start
        #returning the optimal solution to the initial problem
        self.alpha = np.array(sol["x"])
        self.alpha = self.alpha.reshape(self.P)
        #returning the number of iterations of the optimizer
        self.n_iters = sol['iterations']
        #computing the train accuracy
        self._train_accuracy(self.alpha)
        #building the sub-sets R and S 
        R,S = self._return_RS()
        #computing the gradient
        grad = self._get_gradient(Q, self.alpha, alpha_zero)
        #returning m and M
        m, M = self._get_minmax(R, S, grad)
        #storing the difference of m and M
        self.diff = m - M
        # computing the objective function
        self.obj_fun = ( (0.5 * (self.alpha.T @ Q @ self.alpha) ) - (self.alpha @ np.ones(self.P)) )
        self._train_accuracy(self.alpha)
        return self.alpha
    
    def _train_accuracy(self,alpha):
        """
        Compute the train_accuracy.
        :return void
        """
        #reshaping alpha according to P 
        alpha = alpha.reshape(self.P)
        #setting threshold to filter out the optimal alpha
        threshold = 1e-5
        sv = ((alpha > threshold) * (alpha < self.C)).flatten()
        #computing the bias
        self.b = np.mean(self.y[sv] - ((alpha[sv] * self.y[sv]) @ self.K[sv, sv]))
        #labels predictions
        y_predict = (alpha[sv] * self.y[sv]) @ SVM.Kernel(self.x, self.x[sv], self.gamma).T
        #classification function 
        self.train_y_pred = np.sign(y_predict + self.b)
        self.train_accuracy = sum(self.train_y_pred == self.y) / len(self.y)
        
    def predict(self, alpha):
        
        """
        Compute the test_accuracy.
        :return void
        """
        #reshaping alpha according to P 
        alpha = alpha.reshape(self.P)
        #setting threshold to filter out the optimal alpha
        threshold = 1e-5
        sv = ((alpha > threshold) * (alpha < self.C)).flatten()
        #computing the bias
        self.b = np.mean(self.y[sv] - ((alpha[sv] * self.y[sv]) @ self.K[sv, sv]))
        #labels predictions
        y_predict = (alpha[sv] * self.y[sv]) @ SVM.Kernel(self.x_test, self.x[sv], self.gamma).T
        #classification function
        self.y_pred = np.sign(y_predict + self.b)
        self.test_accuracy = sum(self.y_pred == self.y_test) / len(self.y_test)

    def multilabels_predict(self,xtest,alpha):
        
        """
        Compute the test_accuracy.
        :return void
        """
        #reshaping alpha according to P 
        alpha = alpha.reshape(self.P)
        #setting threshold to filter out the optimal alpha
        threshold = 1e-5
        sv = ((alpha > threshold) * (alpha < self.C)).flatten()
        #computing the bias
        self.b = np.mean(self.y[sv] - ((alpha[sv] * self.y[sv]) @ self.K[sv, sv]))
        #labels predictions
        y_predict = (alpha[sv] * self.y[sv]) @ SVM.Kernel(xtest, self.x[sv], self.gamma).T + self.b
        return y_predict
    