# tsv reading part
import math
import numpy as np
from matplotlib import pyplot as plt

"""
This function creates a dictionary according to the length of the input numpy array.  
"""
def createDic(my_np_array):
    dic = {}
    for i in range(len(my_np_array)):
        if int(my_np_array[i][0]) in dic:  # if the key already exists, append another element to the key
            dic[int(my_np_array[i][0])].append(my_np_array[i][1:])
        else:  # if key doesn't exist, create a new key, but have values of dict as a list because we will append more values
            dic[int(my_np_array[i][0])] = [my_np_array[i][1:]]
    return dic

"""
This function returns the number of the first relevant results given a query, the search engine configuration and 
the ground truth. 
"""
def first_relevant_result(q, se_d, gt_d):
    for i in range(len(se_d[q])):
        for j in range(len(gt_d[q])):
            if se_d[q][i][0] == gt_d[q][j]:
                return i + 1
    return 0

"""
MRR (Mean Reciprocal Rank) Function
Input parameters: 
- Q = list of the queries
- se_d = search engine configuration
- ground truth
"""
def MRR(Q, se_d, gt_d):
    # Q: list of the queries
    summation = 0
    for q in Q:
        if not first_relevant_result(q, se_d, gt_d) == 0:
            summation += 1 / (first_relevant_result(q, se_d, gt_d))

    summation *= 1 / len(Q)
    return summation

"""
R Precision function
- input parameters: search engine configuration, ground truth and the query.
"""
def R_precision(se_d, gt_d, q):
    # Formulation : # of relevant docs in first |GT(q)| positions / |GT(q)|
    num = 0
    for i in range(len(gt_d[q])):
        if se_d[q][i][0] in gt_d[q]:
            num += 1
    den = len(gt_d[q])
    return num / den

"""
R_precision_distribution function collects R_precision value for each ground truth query id
- input parameters: search engine configuration, ground truth
This function considers only the query that has ground truth value
"""
def R_precision_distribution(se_d, gt_d):
    my_dist_array = np.zeros(len(gt_d.keys()))

    for i, j in zip(gt_d.keys(), range(len(gt_d.keys()))):
        my_dist_array[j] = R_precision(se_d, gt_d, i)

    return my_dist_array

"""
Precision at K function
Input parameters: search engine configuration, ground truth, query and the integer number of k. 
In this function we implemented the correct version of the P@k.
the one that takes into account also the cardinality of the ground truth(this allows to reach the maximum value of this
evaluation function). 
"""
def P_at_k(se_d, gt_d, q, k):
    # Formulation : # of relevant docs in first k positions / min(|GT(q)|,k)
    num = 0
    for i in range(k):
        if se_d[q][i][0] in gt_d[q]:
            num += 1
    den = min(len(gt_d[q]), k)
    return num / den


def P_at_k_for_all(se_d, gt_d, k):
    result = np.zeros(len(gt_d.keys()))

    for i, j in zip(range(len(gt_d.keys())), gt_d.keys()):
        result[i] = P_at_k(se_d, gt_d, j, k)
    return result

"""
Discounted Cumulative Gain
input parameters: query, k (number of elements considered in the evaluation), ground truth. 
"""
def DCG(q, k, se_d, gt_d):
    result = 0
    for i in range(k):
        if se_d[q][i][0] in gt_d[q]:
            result += 1 / math.log2(2 + i)
    return result

"""
Ideal Discounted Cumulative Gain: 
This is the ideal version of the DCG and is is the result that an ideal search engine would provide.
"""
def IDCG(q, k, gt_d):
    result = 0
    minimum = min(len(gt_d[q]), k)
    for i in range(minimum):
        result += 1 / math.log2(2 + i)
    return result

"""
Normalized Discounted Cumulative Gain
"""
def nDCG(q, k, se_d, gt_d):
    # Formulation nDCG(q, k) = DCG(q,k)/IDCG(q,k)
    # IDCG(q,k) is the DCG(q, k) of a perfect ranking algorithm.
    if not IDCG(q, k, gt_d) == 0:
        result = DCG(q, k, se_d, gt_d) / IDCG(q, k, gt_d)
        return result
    else:
        return 0

def nDCG_for_all(k, se_d, gt_d):
    result = []
    sub_result = []
    for i in gt_d.keys():
        sub_result.append(nDCG(i, k, se_d, gt_d))
    result.append(sum(sub_result) / len(sub_result))
    return np.array(result)


# 1st step : tsv to numpy array

se1 = np.loadtxt(fname="./part_1/part_1_2/dataset/part_1_2__Results_SE_1.tsv", delimiter="\t", skiprows=1)
se2 = np.loadtxt(fname="./part_1/part_1_2/dataset/part_1_2__Results_SE_2.tsv", delimiter="\t", skiprows=1)
se3 = np.loadtxt(fname="./part_1/part_1_2/dataset/part_1_2__Results_SE_3.tsv", delimiter="\t", skiprows=1)
GT = np.loadtxt(fname="./part_1/part_1_2/dataset/part_1_2__Ground_Truth.tsv", delimiter="\t", skiprows=1)


# 2nd step: creating dictionary; query id as key, [doc id, rank, score] as value
# The aim of this step is to decrease computational time

se1_d = createDic(se1)
se2_d = createDic(se2)
se3_d = createDic(se3)
gt_d = createDic(GT)

all_se = [se1_d, se2_d, se3_d]

# Mean Reciprocal Rank(MRR)
MRR_score_list = np.zeros(len(all_se))

for i, j in zip(range(len(all_se)), all_se):
    MRR_score_list[i] = MRR(gt_d.keys(), j, gt_d)

print(MRR_score_list)

# R precision
for i, j in zip(range(len(all_se)), all_se):
    RPD = R_precision_distribution(j, gt_d)
    print('SE', str(i + 1))
    print("mean", "min", "1st q.", "median", "3rd q.", "max", sep='\t')
    print(round(np.mean(RPD), 5), round(min(RPD), 5), round(np.quantile(RPD, .25), 5), round(np.quantile(RPD, .5), 5),
          round(np.quantile(RPD, .75), 5), round(max(RPD), 5), sep='\t')


# Average p@k graph
k = [1,2,3,4]
plt.figure()
for i, t in zip([0,1,2], range(3)):
    x_axis = k
    y_axis = []
    for j in k:
        y_axis.append(np.mean(P_at_k_for_all(all_se[i], gt_d, j)))
    y_axis = np.round(y_axis, 5)
    plt.style.use('ggplot')
    plt.xlabel("k values")
    plt.ylabel("Average p@k")
    plt.plot(x_axis, y_axis, label = 'SE'+str(t+1))
    for a, b in zip(x_axis, y_axis):
        plt.annotate(str(b), xy=(a, b), fontsize = 6)
    plt.legend()
plt.show()

# nDCG@k plot
plt.figure()
for i, t in zip([0,1,2], range(3)):
    x_axis = k
    y_axis = []
    for j in k:
        y_axis.append(np.mean(nDCG_for_all(j, all_se[i], gt_d)))
    y_axis = np.round(y_axis, 5)
    plt.style.use('ggplot')
    plt.xlabel("k values")
    plt.ylabel("Average nDCG@k")
    plt.plot(x_axis, y_axis, label = 'SE'+str(t+1))
    for a, b in zip(x_axis, y_axis):
        plt.annotate(str(b), xy=(a, b), fontsize = 6)
    plt.legend()
plt.show()