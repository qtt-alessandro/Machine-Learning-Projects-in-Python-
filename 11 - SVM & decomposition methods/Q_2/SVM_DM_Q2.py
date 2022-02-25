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


class SVM_DM(object): 

# Static Methods
    @staticmethod
    def Kernel(x1, x2, gamma):
        '''
        Gaussian Kernel.
        :return G.K. matrix -> (x1_dim_0, x2_dim_0)
        '''
        diff = np.expand_dims(x1,axis=2) - x2.T # (Px1, n_features, Px2)
        return np.exp(-gamma * np.sum(diff**2, axis=1)) # (Px1, Px2)

# Object methods
    def __init__(self, x, y, x_test, y_test, C=1, gamma=0.1, q=10):

        # data
        self.x = x
        self.P, _ = x.shape
        self.y = y 
        self.x_test = x_test
        self.y_test =y_test
        # hyper parameters
        self.C = C
        self.gamma = gamma

        if q%2==0:
            self.qq=q
        else:
            print("q not valid! q equal to 10 is selcted.")
            self.qq=10 # default

        self.max_iter=60 # increasing the max_iter do not improve the accuracy of the model 
        # tollerances
        self.tol=1e-3
        self.diff_threshold=1e-6
        # additional variables to print
        self.n_iters=0
        self.elapsed_time = 0
        self.test_accuracy = 0
        self.train_accuracy = 0
        self.obj_fun = 0
        self.n_iters_solver = 0

        # global
        self.K= SVM_DM.Kernel(self.x,self.x,self.gamma)
        self.e = np.ones(self.P)

    ### Fit functions
    def _train_accuracy(self, alpha):
        """
        Compute the train_accuracy.
        :return void
        """
        # setting threshold to filter out the optimal alpha
        threshold = 1e-5
        sv = ((alpha > threshold) * (alpha < self.C)).flatten()
        # computing the bias
        self.b = np.mean(self.y[sv] - ((alpha[sv] * self.y[sv]) @ self.K[sv, sv]))
        # labels predictions
        y_predict = (alpha[sv] * self.y[sv]) @ SVM_DM.Kernel(self.x, self.x[sv], self.gamma).T
        # classification function 
        self.train_y_pred = np.sign(y_predict + self.b)
        self.train_accuracy = sum(self.train_y_pred == self.y) / len(self.y) 

    def _get_RS(self, alpha):
        """
        1. creating the binary vectors T/F: a<C, a>0, y==1, y==-1
        2. according to the theory, create R and S

        :param alpha: vector of length P

        :return the two sets R and S
        """
        a_less_C = alpha < self.C - self.tol
        a_greater_zero = alpha > self.tol
        y_equals_1 = self.y == 1
        y_equals_1 = y_equals_1.reshape(a_greater_zero.shape)
        y_equals_minus_1 = self.y == -1
        y_equals_minus_1 = y_equals_minus_1.reshape(a_greater_zero.shape)
        R_mask = (a_less_C & y_equals_1) | (a_greater_zero & y_equals_minus_1)
        R = np.array([i for i, x in enumerate(R_mask) if x])
        S_mask = (a_less_C & y_equals_minus_1) | (a_greater_zero & y_equals_1)
        S = np.array([i for i, x in enumerate(S_mask) if x])

        return R, S

    def _get_working_set(self,R,S,grad):
        """
        1. sort the vector "-1 * grad * y" for S and R
        2. take into account the first q elements

        :return the Working Set of the k-th iteration
        """
        W=[]
        A = sorted(zip(-1 * grad[R] * self.y[R], R), key = lambda x: x[0], reverse=True) # max to min
        B = sorted(zip(-1 * grad[S] * self.y[S], S), key = lambda x: x[0]) # min to max

        for i,j in zip(A, B):
            W.append(i[1])
            W.append(j[1])
            if len(W)==self.qq:
                break   

        return np.array(W)
    
    def _get_gradient(self, grad, Q, alpha, old_alpha, W):
        """
        :return the updated gradient of the k-th iteration
        """
        return grad + Q[:,W] @ (alpha[W]-old_alpha[W])  

    def _get_minmax(self,R,S,grad):
        """
        :return m,M
        """
        m = np.max(-1 * grad[R] * self.y[R])
        M = np.min(-1 * grad[S] * self.y[S])
        return m,M

    ### Fit 
    def fit(self, show_progress=False):

        # work space and initial parameters
        Q = (np.outer(self.y, self.y) * self.K) # (P,P)
        alpha = np.zeros(self.P) 
        grad = -1 * np.ones(self.P)

        k,m,M,diff=0,0,0,1

        ### CVOPT Parameters ###
        ones = np.ones(self.qq)
        # G (2q,q)
        block_1_G = -np.eye(self.qq)
        block_2_G = np.eye(self.qq)
        G_tmp = np.concatenate((block_1_G,block_2_G), axis=0)
        G = cvxopt.matrix(G_tmp)
        # h (2q,1)
        block_1_h = np.zeros((self.qq))
        block_2_h = self.C*np.ones((self.qq))
        h_tmp = np.concatenate((block_1_h,block_2_h),axis=0)
        h = cvxopt.matrix(h_tmp)  

        if show_progress:
            cvxopt.solvers.options["show_progress"] = True
        else:
            cvxopt.solvers.options["show_progress"] = False

        while(diff>self.diff_threshold and k<self.max_iter):
            
            old_alpha = np.copy(alpha)

            ### selecting the working set ###
            R, S = self._get_RS(alpha) 
            W = self._get_working_set(R,S,grad) 
            N = list(set(np.arange(self.P)) - set(W))

            ### CVOPT Parameters ###
            # P -> Q_ww
            P = cvxopt.matrix(Q[W,:][:,W]) # (q,q)
            # q.T -> alpha^k_N * Q_NW - e.T_W
            q = cvxopt.matrix((alpha[N].T @ Q[N,:][:,W]) - ones) # (q,)
            # A = y.T_W
            A = cvxopt.matrix(self.y[W], (1,self.qq), "d") 
            # Ax=b -> (1,q)@(q,1)-> 1,1
            # b = -y.T_N * alpha^k_N
            b = -1 * self.y[N].T @ alpha[N]
            b = cvxopt.matrix(b) # scalar


            ### CVOPT Solver ###
            start = time.time()
            sol = cvxopt.solvers.qp(P, q, G, h, A, b)
            end = time.time()
            self.n_iters_solver += sol['iterations']
            self.elapsed_time += end-start
            new_alphas = np.array(sol["x"])

            ### updates of the alphas ###
            for i,j in zip(W,new_alphas):
                alpha[i]=j
 
            ### compute the new gradient ###
            grad = self._get_gradient(grad, Q, alpha, old_alpha,W)
            ### compute the difference between the m and M
            m,M=self._get_minmax(R,S,grad)
            diff = m-M

            ### Print
            #print(f"---Iteration no. {k+1}---")
            #print(f'W: {W}')
            #print(new_alphas)
            #print(alpha[W])
            #print(grad)
            #print(diff)

            k=k+1
        else:
            self.obj_fun = ( (0.5 * (alpha.T @ Q @ alpha) ) - (alpha @ self.e) )
            self.n_iters=k
            self.diff=diff
            # print(self.obj_fun)
            # print(self.obj_fun[-1])
            # print(f'number of optimization iterations: {k}')
            # print(f'difference between m(alpha) and M(alpha): {diff}')
            # print(f'elapsed_time: {self.elapsed_time}')

        self._train_accuracy(alpha)

        return alpha

    ### Prediction
    def predict(self, alpha):
        """
        Compute the test_accuracy.
        :return void
        """
        # setting threshold to filter out the optimal alpha
        threshold = 1e-5
        sv = ((alpha > threshold) * (alpha < self.C)).flatten()
        # computing the bias
        self.b = np.mean(self.y[sv] - ((alpha[sv] * self.y[sv]) @ self.K[sv, sv]))
        # labels predictions
        y_predict = (alpha[sv] * self.y[sv]) @ SVM_DM.Kernel(self.x_test, self.x[sv], self.gamma).T
        # classification function
        self.y_pred = np.sign(y_predict + self.b)
        self.test_accuracy = sum(self.y_pred == self.y_test) / len(self.y_test)