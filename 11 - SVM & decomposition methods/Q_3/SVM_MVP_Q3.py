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



class SVM_MVP(object):
    def __init__(self, x, y, x_test, y_test, C=1, gamma=0.1, max_iter=2500):

        self.x = x
        self.y = y
        self.x_test = x_test
        self.y_test = y_test
        self.P, _ = x.shape
        self.C = C
        self.e = np.ones(self.P)
        self.gamma = gamma
        self.max_iter = 2500
        self.epsilon = 1e-4
        self.elapsed_time = 0
        self.n_iters = 0
        self.K = SVM_MVP.Kernel(x,x,gamma)
        self.Q = np.outer(y, y) * self.K
        self.y_pred = 0
        self.test_accuracy = 0
        self.train_accuracy = 0
        self.diff_threshold = 1e-4
        self.kkt_viol = 0
        self.obj_fun = 0
    
    @staticmethod
    def Kernel(x, c, gamma):
        '''
        Gaussian Kernel.
        :return output -> (P,P)
        '''
        diff = np.expand_dims(x, axis=2) - c.T  # (Px, 16, Pc)
        return np.exp(-gamma * np.sum(diff ** 2, axis=1))  # (Px, Pc)


    def fit(self):
        start = time.time()
        # Initial alpha values are fixed to 0.
        alpha = np.zeros((self.P, 1))
        new_alpha = alpha.copy()
        # Initial gradient is always taken as -1 vector.
        gradient = -self.e
        # First step is to choose R and S space
        R, S = self.return_R_S(alpha)
        # Second step is to choose i and j that gives the most violating pair(MVP)
        i, j = self.return_I_J(R, S, gradient)
        # We calculate the m(alpha) - M(alpha) to check if it satisfy the convergence/KKT condition
        diff = -self.y[i] * gradient[i] + self.y[j] * gradient[j]
        # If the difference is still not reached, we can consider a maximum iteration bound to stop calculation
        while (self.max_iter > self.n_iters and diff > self.epsilon):
            # This step uses analytic solver, since we can use it when we apply MVP(q=2)
            alpha_opt = self.analytic_solver(i, j, gradient, self.Q, alpha)
            # The following steps are important to check alpha boundary conditions:
            # If the alpha is less than self.epsilon or greater than self.C - self.epsilon, we will equal them to
            # self.epsilon and self.C - self.epsilon respectively.
            if alpha_opt[0] < self.epsilon:
                alpha_opt[0] = self.epsilon
            elif alpha_opt[0] > self.C - self.epsilon:
                alpha_opt[0] = self.C - self.epsilon
            if alpha_opt[1] < self.epsilon:
                alpha_opt[1] = self.epsilon
            elif alpha_opt[1] > self.C - self.epsilon:
                alpha_opt[1] = self.C - self.epsilon
            # We will not update alpha since we need previous alpha and new alpha to calculate gradient.
            new_alpha[i] = alpha_opt[0]
            new_alpha[j] = alpha_opt[1]
            gradient = self.calculate_gradient(alpha, new_alpha, gradient, self.Q, [i, j])
            alpha = new_alpha.copy()
            # Then, the flow continues as before.
            R, S = self.return_R_S(alpha)
            i, j = self.return_I_J(R, S, gradient)
            diff = -self.y[i] * gradient[i] + self.y[j] * gradient[j]
            # The KKT violation is calculated to report later on
            if diff >= 0:
                self.kkt_viol += 1
            self.n_iters += 1
        end = time.time()
        self.elapsed_time = end - start
        self.diff = diff
        self._train_accuracy(alpha)
        self.obj_fun = ((0.5 * (alpha.T @ self.Q @ alpha)) - (alpha.T @ self.e))
        return alpha

    def return_R_S(self, alpha):
        """
        Return the R and S sets that will help decomposition methods to choose violating pairs.
        :param alpha: vector of length P
        :return: R and S sets(numpy vectors) that are candidates for the Decomposition method's violating pairs
        """
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

    def return_I_J(self, R, S, gradient):
        """
        Returns the indexes of the most violating pair
        :param R: numpy array, first output of the return_R_S function
        :param S: numpy array, second output of the return_R_S function
        :param gradient: numpy array of lenght P
        :return: the indexes of the MVP
        """
        max_index = np.argmax(-self.y[R] * gradient[R])
        i = R[max_index]
        min_index = np.argmin(-self.y[S] * gradient[S])
        j = S[min_index]
        return i, j

    def analytic_solver(self, i, j, gradient, Q, alpha):
        """
        Calculates the optimal alpha values of the most violating pair(i, j)
        :param i: first element of the MVP
        :param j: second element of the MVP
        :param gradient: numpy array of lenght P
        :param Q: one of the parameter that has shape of P,P and calculated by using RBF Kernel
        :param alpha: vector of length P
        :return: the optimal alpha values, has lenght of 2 (corresponding to most violating pair)
        """
        alpha_interest = np.array([alpha[i], alpha[j]]).reshape(2, 1)
        w = [i, j]
        Q_interest = Q[w, :][:, w]
        gradient_i_j = np.array([gradient[i], gradient[j]]).reshape(2, 1)
        d_ij = np.array([self.y[i], -self.y[j]]).reshape(2, 1)

        if gradient_i_j.T @ d_ij == 0:
            return alpha_interest
        else:
            if gradient_i_j.T @ d_ij < 0:
                d_opt = d_ij
            elif gradient_i_j.T @ d_ij > 0:
                d_opt = -d_ij

            if d_ij[0] > 0:
                if d_ij[1] > 0:
                    beta_ = min(self.C - alpha[i], self.C - alpha[j])
                elif d_ij[1] < 0:
                    beta_ = min(self.C - alpha[i], alpha[j])
            elif d_ij[0] < 0:
                if d_ij[1] > 0:
                    beta_ = min(alpha[i], self.C - alpha[j])
                elif d_ij[1] < 0:
                    beta_ = min(alpha[i], alpha[j])

            if beta_ == 0:
                beta_opt = 0
            elif d_opt.T @ Q_interest @ d_opt == 0:
                beta_opt = beta_
            else:
                if d_opt.T @ Q_interest @ d_opt > 0:
                    beta_nv = (-gradient_i_j.T @ d_opt) / (d_opt.T @ Q_interest @ d_opt)
                    beta_opt = min(beta_, beta_nv)

            alpha_opt = alpha_interest + beta_opt * d_opt
            return alpha_opt

    def calculate_gradient(self, prev_alpha, current_alpha, gradient, Q, w):
        """
        :param prev_alpha: the previous alpha value
        :param current_alpha: the updated alpha value
        :param gradient: numpy array of lenght P
        :param Q: one of the parameter that has shape of P,P and calculated by using RBF Kernel
        :param w: the working set
        :return: the updated gradient of the k-th iteration
        """
        # w: working set
        result = gradient
        result += np.sum((current_alpha[w] - prev_alpha[w]).T * Q[:, w], axis=1)
        return result

    def _train_accuracy(self, alpha):
        alpha = alpha.reshape(self.P)
        threshold = 1e-5
        sv = ((alpha > threshold) * (alpha < self.C)).flatten()
        self.b = np.mean(self.y[sv] - ((alpha[sv] * self.y[sv]) @ self.K[sv, sv]))

        y_predict = (alpha[sv] * self.y[sv]) @ SVM_MVP.Kernel(self.x, self.x[sv], self.gamma).T
        self.train_y_pred = np.sign(y_predict + self.b)
        self.train_accuracy = sum(self.train_y_pred == self.y) / len(self.y)

    def predict(self, alpha):

        alpha = alpha.reshape(self.P)
        threshold = 1e-5
        sv = ((alpha > threshold) * (alpha < self.C)).flatten()
        self.b = np.mean(self.y[sv] - ((alpha[sv] * self.y[sv]) @ self.K[sv, sv]))

        y_predict = (alpha[sv] * self.y[sv]) @ SVM_MVP.Kernel(self.x_test, self.x[sv], self.gamma).T
        self.y_pred = np.sign(y_predict + self.b)
        self.test_accuracy = sum(self.y_pred == self.y_test) / len(self.y_test)




