from whoosh.index import create_in
from whoosh.fields import *
from whoosh.analysis import StemmingAnalyzer, StandardAnalyzer, SimpleAnalyzer, FancyAnalyzer, StopFilter
import csv
import os, os.path
from whoosh import index
from whoosh.qparser import *
from whoosh import scoring
import pandas as pd
import re
from bs4 import BeautifulSoup
import math
import numpy as np
from matplotlib import pyplot as plt


# HTML Scraping
############################################################################################
""""
We built this function for scraping htm pages and 
stored the results in two different .csv files Cranfield_DATASET.csv and Time_DATASET.csv
"""
def extract_data(DIR, title_exists=False):
    # DIR : Directory to extract the dataset
    os.chdir(DIR)
    # n_d : number of docs in dataset
    n_d = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    docs = os.listdir(DIR)
    if title_exists == True:
        my_array = np.empty((0, 3))
    else:
        my_array = np.empty((0, 2))
    counter = 0
    for doc in docs:
        counter += 1
        result = re.search('______(.*).html', doc)
        doc_id = result.group(1)

        with open(doc) as fp:
            # Use BeautifulSoup to parse html type
            soup = BeautifulSoup(fp, "html.parser")
            body = soup.body.text
            # If title wanted to be extracted title_decision == True
            if title_exists == True:
                title = soup.head.title.text
                my_array = np.append(my_array, np.array([[doc_id, title, body]]), axis=0)
            else:
                my_array = np.append(my_array, np.array([[doc_id, body]]), axis=0)

    DF = pd.DataFrame(my_array)
    # save the dataframe as a csv file
    if title_exists == True:
        DF.to_csv("Cranfield_DATASET.csv", header=['doc_id', 'title', 'content'], index=False)
    else:
        DF.to_csv("Time_DATASET.csv", header=['doc_id', 'content'], index=False)

    return my_array

"""
This is an auxiliary function often used within other 
macro functions to generate a folder given the name and destination address. 
"""
############################################################################################

def create_folder(address,name):
    path = address + "/" + name
    print(path)
    try:
        os.mkdir(path)
    except:
        print("This Folder already exists")
    return path

#PIPELINE: CREATE SCHEMA + FILLING SCHEMA + RANKING
############################################################################################
"""
From this point onwards we report on the three characteristic functions of the pipeline 
for schema creation, schema filling and the ranking with different scoring functions. 
"""

"""
This function creates the empty schema according to the text analyzer
and also create a folder which contains the empty schema. 
Since we have two different dataset, we generalized this function adding a title_Flag. 
Thi has allowed to use the same function for the two different .csv input file.
"""
def create_schema(selected_analyzer, name_analyzer,title_FLAG = True):
    address = './part_1/part_1_1'
    path = create_folder(address,name_analyzer)
    if title_FLAG == True:
        schema = Schema(id=ID(stored=True), \
                        title=TEXT(stored=False, analyzer=selected_analyzer) , \
                        content=TEXT(stored=False, analyzer=selected_analyzer))
        create_in(path, schema)
        print("The schema with title field is created at:",path )
        return path
    else:
        schema = Schema(id=ID(stored=True), \
                        content=TEXT(stored=False, analyzer=selected_analyzer))
        create_in(path, schema)
        print("The schema (without title) is created at:", path)
        return path

"""
This is the part of the pipeline needed for filling the empty schema we created in the first step.
"""
def fill_index(directory_containing_the_index, dataset_path,title_FLAG = True):
    ix = index.open_dir(directory_containing_the_index)
    writer = ix.writer()
    ALL_DOCUMENTS_file_name = dataset_path
    in_file = open(ALL_DOCUMENTS_file_name, "r", encoding='latin1')
    csv_reader = csv.reader(in_file, delimiter=',')
    csv_reader.__next__()
    num_added_records_so_far = 0
    if title_FLAG == True:
        for record in csv_reader:
            id    = record[0]
            title_doc = record[1]
            plot  = record[2]
            #
            writer.add_document(id=id,title=title_doc,content=plot)
            #
            num_added_records_so_far += 1
            if (num_added_records_so_far % 100 == 0):
                print("num_added_records_so_far= " + str(num_added_records_so_far))
        print("writing committing...")
        writer.commit()
        print("closing file...")
        in_file.close()
        print("Index correctly filled")
    else:
        for record in csv_reader:
            id    = record[0]
            plot  = record[1]
            #
            writer.add_document(id=id,content=plot)
            #
            num_added_records_so_far += 1
            if (num_added_records_so_far % 100 == 0):
                print("num_added_records_so_far= " + str(num_added_records_so_far))
        print("writing committing...")
        writer.commit()
        print("closing file...")
        in_file.close()
        print("Index correctly filled")

"""
This is the last part of the pipeline, in facts returns a user fixed number of results ranked according to the 
given the schema and the filled index created by the first two box of the pipeline.
"""
def scoring_function(directory_containing_the_index, name_out_file, folder_output, dataset_folder_address,
                     max_number_of_results, score , title_FLAG = True):

    tsv_file = open(dataset_folder_address)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    next(read_tsv)
    with open(folder_output + '/' + name_out_file + '.tsv', 'w') as f1:
        writer = csv.writer(f1, delimiter='\t', lineterminator='\n')
        writer.writerow(["Query_ID", "Doc_ID", "Rank", "Score"])
        query_count = 0
        for row in read_tsv:
            query_ID = row[0]
            query_corpus = row[1]
            ix = index.open_dir(directory_containing_the_index)
            if title_FLAG == True:
                qp = MultifieldParser(["title", "content"],ix.schema)
            elif title_FLAG == False:
                qp = QueryParser("content", ix.schema)
            parsed_query = qp.parse(query_corpus)
            searcher = ix.searcher(weighting=score)
            results = searcher.search(parsed_query, limit=max_number_of_results)
            query_count += 1
            if (query_count % 10 == 0):
                print(" num_queries_processed_so_far= " + str(query_count))
            if (query_count == 222 or query_count == 83 ):
                print("finished")
            for hit in results:
                # print(str(hit.rank + 1) + "\t" + hit['id'] + "\t" + str(hit.score))
                writer = csv.writer(f1, delimiter='\t', lineterminator='\n')
                writer.writerow([str(row[0]), hit['id'], str(hit.rank + 1), str(hit.score)])
            searcher.close()


"""
This function executes the entire pipeline by combining the three blocks presented above. 
Takes the parameters to compute create_folder, create_schema, fill_index, and scoring function.
Therefore, different Search Engines could be created with using it.
"""
def compose_SE(selected_analyzer, score, address, name_analyzer, dataset_path, max_number_of_results, name_out_file, dataset_folder_address, title_FLAG = True):

    folder_output = address + '/' + name_analyzer
    directory_containing_the_index = create_schema(selected_analyzer=selected_analyzer, \
                                                   name_analyzer=name_analyzer, \
                                                   title_FLAG=title_FLAG)
    fill_index(directory_containing_the_index=directory_containing_the_index, \
               dataset_path=dataset_path, \
               title_FLAG=title_FLAG)
    scoring_function(directory_containing_the_index=directory_containing_the_index, \
                     name_out_file=name_out_file, \
                     folder_output=folder_output, \
                     dataset_folder_address=dataset_folder_address, \
                     max_number_of_results=max_number_of_results, \
                     score=score)

############################################################################################

"""
This function creates a dictionary according to the length of the input numpy array.  
"""
def createDic(my_np_array):
    dic = {}
    for i in range(len(my_np_array)):
        if int(my_np_array[i][0]) in dic:  # if the key already exists, append another element to the key
            dic[int(my_np_array[i][0])].append(my_np_array[i][1:])
        # if key doesn't exist, create a new key, but have values of dict as a list because we will append more values
        else:
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

#EVALUATION METRICS FUNCTIONS
############################################################################################
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
    summation *= (1 / len(Q))
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
        result += 1/math.log2(2 + i)
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
    result.append(sum(sub_result)/len(gt_d.keys()))
    return np.array(result)
############################################################################################


cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))

# CALLING SCRAPING FUNCTIONS FOR CRAN AND TIME DATASET
############################################################################################
DIR = str(cwd + '/part_1/part_1_1/dataset/Cranfield_DATASET/DOCUMENTS')
cran_data = extract_data(DIR, True)

DIR = str(cwd + '/part_1/part_1_1/dataset/Time_DATASET/DOCUMENTS')
time_data = extract_data(DIR, False)
############################################################################################




"""
Some useful variables used as input for the following functions
"""
dataset_path_Cran = "./part_1/part_1_1/dataset/Cranfield_DATASET/DOCUMENTS/Cranfield_DATASET.csv"
dataset_path_Time = "./part_1/part_1_1/dataset/Time_DATASET/DOCUMENTS/Time_DATASET.csv"
queries_data_path_Cran = "./part_1/part_1_1/dataset/Cranfield_DATASET/cran_Queries.tsv"
queries_data_path_Time = "./part_1/part_1_1/dataset/Time_DATASET/time_Queries.tsv"
folder_output_dir = './part_1/part_1_1'



#################################CRANFIELD DATASET(up to 495th line)######################
##########################1.SIMPLE ANALYZER_TFID #########################################
compose_SE(SimpleAnalyzer(), scoring.TF_IDF(), folder_output_dir, 'Simple_TFIDF',
           dataset_path_Cran, 40, 'SE1_results',queries_data_path_Cran)
##########################2.SIMPLE ANALYZER_BM25F #########################################
compose_SE(SimpleAnalyzer(), scoring.BM25F(B=0.75, content_B=1.0, K1=1.5), folder_output_dir, 'Simple_BM25F',
           dataset_path_Cran, 40, 'SE2_results',queries_data_path_Cran)
##########################3.STANDARD ANALYZER WITH STOP FILTER_TFID #########################################
compose_SE(StandardAnalyzer() | StopFilter(), scoring.TF_IDF(), folder_output_dir, 'Std_Stop_TFIDF',
           dataset_path_Cran, 40, 'SE3_results',queries_data_path_Cran)
##########################4.STANDARD ANALYZER WITH STOP FILTER_BM25F #######################
compose_SE(StandardAnalyzer() | StopFilter(), scoring.BM25F(B=0.75, content_B=1.0, K1=1.5),folder_output_dir, 'Std_Stop_BM25F',
           dataset_path_Cran, 40, 'SE4_results',queries_data_path_Cran)
##########################5.STEMMING ANALYZER_TFIDF#########################################
compose_SE(StemmingAnalyzer(), scoring.TF_IDF(),folder_output_dir, 'Stem_TFIDF',
           dataset_path_Cran, 40, 'SE5_results',queries_data_path_Cran)
##########################6.STEMMING ANALYZER_BM25F#########################################
compose_SE(StemmingAnalyzer(), scoring.BM25F(B=0.75, content_B=1.0, K1=1.5),folder_output_dir, 'Stem_BM25F',
           dataset_path_Cran, 40, 'SE6_results',queries_data_path_Cran)
##########################7.STANDARD_TDFIDF#####################
compose_SE(StandardAnalyzer(), scoring.TF_IDF(),folder_output_dir, 'Std_TFIDF',
           dataset_path_Cran, 40, 'SE7_results',queries_data_path_Cran)
##########################8.FANCY ANALYZER_TFIDF###########################################
compose_SE(FancyAnalyzer(), scoring.TF_IDF(),folder_output_dir, 'Fancy_TFIDF',
           dataset_path_Cran, 40, 'SE8_results',queries_data_path_Cran)
##########################9.FANCY ANALYZER_BM25F############################################
compose_SE(FancyAnalyzer(), scoring.BM25F(B=0.75, content_B=1.0, K1=1.5),folder_output_dir, 'Fancy_BM25F',
           dataset_path_Cran, 40, 'SE9_results',queries_data_path_Cran)
##########################10.FANCY ANALYZER_FREQUENCY########################################
compose_SE(FancyAnalyzer(), scoring.Frequency(),folder_output_dir, 'Fancy_Frequency',
           dataset_path_Cran, 40, 'SE10_results',queries_data_path_Cran)

################################################################################################

# EVALUATION METRICS TIME DATASET
############################################################################################
############################################################################################
# 1st step : tsv to numpy array
se1 = np.loadtxt(fname="./part_1/part_1_1/Simple_TFIDF/SE1_results.tsv", delimiter="\t", skiprows=1)
se2 = np.loadtxt(fname="./part_1/part_1_1/Simple_BM25F/SE2_results.tsv", delimiter="\t", skiprows=1)
se3 = np.loadtxt(fname="./part_1/part_1_1/Std_Stop_TFIDF/SE3_results.tsv", delimiter="\t", skiprows=1)
se4 = np.loadtxt(fname="./part_1/part_1_1/Std_Stop_BM25F/SE4_results.tsv", delimiter="\t", skiprows=1)
se5 = np.loadtxt(fname="./part_1/part_1_1/Stem_TFIDF/SE5_results.tsv", delimiter="\t", skiprows=1)
se6 = np.loadtxt(fname="./part_1/part_1_1/Stem_BM25F/SE6_results.tsv", delimiter="\t", skiprows=1)
se7 = np.loadtxt(fname="./part_1/part_1_1/Std_TFIDF/SE7_results.tsv", delimiter="\t", skiprows=1)
se8 = np.loadtxt(fname="./part_1/part_1_1/Fancy_TFIDF/SE8_results.tsv", delimiter="\t", skiprows=1)
se9 = np.loadtxt(fname="./part_1/part_1_1/Fancy_BM25F/SE9_results.tsv", delimiter="\t", skiprows=1)
se10 = np.loadtxt(fname="./part_1/part_1_1/Fancy_Frequency/SE10_results.tsv", delimiter="\t", skiprows=1)
GT = np.loadtxt(fname="./part_1/part_1_1/dataset/Cranfield_DATASET/cran_Ground_Truth.tsv", delimiter="\t", skiprows=1)

# 2nd step: creating dictionary; query id as key, [doc id, rank, score] as value
# The aim of this step is to decrease computational time
se1_d = createDic(se1)
se2_d = createDic(se2)
se3_d = createDic(se3)
se4_d = createDic(se4)
se5_d = createDic(se5)
se6_d = createDic(se6)
se7_d = createDic(se7)
se8_d = createDic(se8)
se9_d = createDic(se9)
se10_d = createDic(se10)
gt_d = createDic(GT)

# We created 10 different Search Engines and these will be used in the upcoming functions
all_se = [se1_d, se2_d, se3_d, se4_d, se5_d, se6_d, se7_d, se8_d, se9_d, se10_d]
# It is given that k values are the below
k = np.array([1, 3, 5, 10])

# Mean Reciprocal Rank(MRR)

# MRR_score_list is created to choose top 5 SEs
MRR_score_list = np.zeros(len(all_se))

for i, j in zip(range(len(all_se)), all_se):
    MRR_score_list[i] = MRR(gt_d.keys(), j, gt_d)

print(MRR_score_list)
# Indexes of top 5 is stored in index_top_5 variable
index_top_5 = MRR_score_list.argsort()[-5:][::-1]


# R precision

for i, j in zip(range(len(all_se)), all_se):
    RPD = R_precision_distribution(j, gt_d)
    print('SE', str(i + 1))
    print("mean", "min", "1st q.", "median", "3rd q.", "max", sep='\t')
    print(round(np.mean(RPD), 5), round(min(RPD), 5), round(np.quantile(RPD, .25), 5), round(np.quantile(RPD, .5), 5),
          round(np.quantile(RPD, .75), 5), round(max(RPD), 5), sep='\t')

# P@k plot for the best 5 SEs

# P@k
# Average p@k graph

plt.figure()
for i, t in zip(index_top_5, range(len(all_se))):
    x_axis = k
    y_axis = []
    for j in k:
        y_axis.append(np.mean(P_at_k_for_all(all_se[i], gt_d, j)))
    y_axis = np.round(y_axis, 5)
    plt.style.use('ggplot')
    plt.xlabel("k values")
    plt.ylabel("Average p@k")
    plt.plot(x_axis, y_axis, label = 'SE'+str(index_top_5[t]+1))
    for a, b in zip(x_axis, y_axis):
        plt.annotate(str(b), xy=(a, b), fontsize = 6)
    plt.legend()
plt.show()

# nDCG@k plot for top 5 SEs

# Normalized Discounted Cumulative Gain: nDCG(q,k)

plt.figure()
for i, t in zip(index_top_5, range(5)):
    x_axis = k
    y_axis = []
    for j in k:
        y_axis.append(np.mean(nDCG_for_all(j, all_se[i], gt_d)))
    y_axis = np.round(y_axis, 5)
    plt.style.use('ggplot')
    plt.xlabel("k values")
    plt.ylabel("Average nDCG@k")
    plt.plot(x_axis, y_axis, label = 'SE'+str(index_top_5[t]+1))
    for a, b in zip(x_axis, y_axis):
        plt.annotate(str(b), xy=(a, b), fontsize = 6)
    plt.legend()
plt.show()
############################################################################################
############################################################################################

#################################TIME DATASET(up to 495th line)###########################
##########################1.SIMPLE ANALYZER_TFID #########################################
compose_SE(SimpleAnalyzer(), scoring.TF_IDF(), folder_output_dir, 'Simple_TFIDF_Time',
           dataset_path_Time, 40, 'SE1_results',queries_data_path_Time, title_FLAG=False)
##########################2.SIMPLE ANALYZER_BM25F #########################################
compose_SE(SimpleAnalyzer(), scoring.BM25F(B=0.75, content_B=1.0, K1=1.5), folder_output_dir, 'Simple_BM25F_Time',
           dataset_path_Time, 40, 'SE2_results',queries_data_path_Time, title_FLAG=False)
##########################3.STANDARD ANALYZER WITH STOP FILTER_TFID #########################################
compose_SE(StandardAnalyzer() | StopFilter(), scoring.TF_IDF(), folder_output_dir, 'Std_Stop_TFIDF_Time',
           dataset_path_Time, 40, 'SE3_results',queries_data_path_Time, title_FLAG=False)
##########################4.STANDARD ANALYZER WITH STOP FILTER_BM25F #######################
compose_SE(StandardAnalyzer() | StopFilter(), scoring.BM25F(B=0.75, content_B=1.0, K1=1.5),folder_output_dir, 'Std_Stop_BM25F_Time',
           dataset_path_Time, 40, 'SE4_results',queries_data_path_Time, title_FLAG=False)
##########################5.STEMMING ANALYZER_TFIDF#########################################
compose_SE(StemmingAnalyzer(), scoring.TF_IDF(),folder_output_dir, 'Stem_TFIDF_Time',
           dataset_path_Time, 40, 'SE5_results',queries_data_path_Time, title_FLAG=False)
##########################6.STEMMING ANALYZER_BM25F#########################################
compose_SE(StemmingAnalyzer(), scoring.BM25F(B=0.75, content_B=1.0, K1=1.5),folder_output_dir, 'Stem_BM25F_Time',
           dataset_path_Time, 40, 'SE6_results',queries_data_path_Time, title_FLAG=False)
##########################7.STANDARD_TDFIDF#####################
compose_SE(StandardAnalyzer(), scoring.TF_IDF(),folder_output_dir, 'Std_TFIDF_Time',
           dataset_path_Time, 40, 'SE7_results',queries_data_path_Time, title_FLAG=False)
##########################8.FANCY ANALYZER_TDFIDF###########################################
compose_SE(FancyAnalyzer(), scoring.TF_IDF(),folder_output_dir, 'Fancy_TFIDF_Time',
           dataset_path_Time, 40, 'SE8_results',queries_data_path_Time, title_FLAG=False)
##########################9.FANCY ANALYZER_BM25F############################################
compose_SE(FancyAnalyzer(), scoring.BM25F(B=0.75, content_B=1.0, K1=1.5),folder_output_dir, 'Fancy_BM25F_Time',
           dataset_path_Time, 40, 'SE9_results',queries_data_path_Time, title_FLAG=False)
##########################10.FANCY ANALYZER_FREQUENCY########################################
compose_SE(FancyAnalyzer(), scoring.Frequency(),folder_output_dir, 'Fancy_Frequency_Time',
           dataset_path_Time, 40, 'SE10_results',queries_data_path_Time, title_FLAG=False)
################################################################################################
################################################################################################

# EVALUATION METRICS
############################################################################################
############################################################################################
# 1st step : tsv to numpy array

se1 = np.loadtxt(fname="./part_1/part_1_1/Simple_TFIDF_Time/SE1_results.tsv", delimiter="\t", skiprows=1)
se2 = np.loadtxt(fname="./part_1/part_1_1/Simple_BM25F_Time/SE2_results.tsv", delimiter="\t", skiprows=1)
se3 = np.loadtxt(fname="./part_1/part_1_1/Std_Stop_TFIDF_Time/SE3_results.tsv", delimiter="\t", skiprows=1)
se4 = np.loadtxt(fname="./part_1/part_1_1/Std_Stop_BM25F_Time/SE4_results.tsv", delimiter="\t", skiprows=1)
se5 = np.loadtxt(fname="./part_1/part_1_1/Stem_TFIDF_Time/SE5_results.tsv", delimiter="\t", skiprows=1)
se6 = np.loadtxt(fname="./part_1/part_1_1/Stem_BM25F_Time/SE6_results.tsv", delimiter="\t", skiprows=1)
se7 = np.loadtxt(fname="./part_1/part_1_1/Std_TFIDF_Time/SE7_results.tsv", delimiter="\t", skiprows=1)
se8 = np.loadtxt(fname="./part_1/part_1_1/Fancy_TFIDF_Time/SE8_results.tsv", delimiter="\t", skiprows=1)
se9 = np.loadtxt(fname="./part_1/part_1_1/Fancy_BM25F_Time/SE9_results.tsv", delimiter="\t", skiprows=1)
se10 = np.loadtxt(fname="./part_1/part_1_1/Fancy_Frequency_Time/SE10_results.tsv", delimiter="\t", skiprows=1)
GT = np.loadtxt(fname="./part_1/part_1_1/dataset/Time_DATASET/time_Ground_Truth.tsv", delimiter="\t", skiprows=1)

se1_d = createDic(se1)
se2_d = createDic(se2)
se3_d = createDic(se3)
se4_d = createDic(se4)
se5_d = createDic(se5)
se6_d = createDic(se6)
se7_d = createDic(se7)
se8_d = createDic(se8)
se9_d = createDic(se9)
se10_d = createDic(se10)
gt_d = createDic(GT)

# We created 10 different Search Engines and these will be used in the upcoming functions
all_se = [se1_d, se2_d, se3_d, se4_d, se5_d, se6_d, se7_d, se8_d, se9_d, se10_d]

# Mean Reciprocal Rank(MRR)

# MRR_score_list is created to choose top 5 SEs
MRR_score_list = np.zeros(len(all_se))

for i, j in zip(range(len(all_se)), all_se):
    MRR_score_list[i] = MRR(gt_d.keys(), j, gt_d)

print(MRR_score_list)
# Indexes of top 5 is stored in index_top_5 variable
index_top_5 = MRR_score_list.argsort()[-5:][::-1]

# R precision
for i, j in zip(range(len(all_se)), all_se):
    RPD = R_precision_distribution(j, gt_d)
    print('SE', str(i + 1))
    print("mean", "min", "1st q.", "median", "3rd q.", "max", sep='\t')
    print(round(np.mean(RPD), 5), round(min(RPD), 5), round(np.quantile(RPD, .25), 5), round(np.quantile(RPD, .5), 5),
          round(np.quantile(RPD, .75), 5), round(max(RPD), 5), sep='\t')


# Average p@k graph
plt.figure()
for i, t in zip(index_top_5, range(5)):
    x_axis = k
    y_axis = []
    for j in k:
        y_axis.append(np.mean(P_at_k_for_all(all_se[i], gt_d, j)))
    y_axis = np.round(y_axis, 5)
    plt.style.use('ggplot')
    plt.xlabel("k values")
    plt.ylabel("Average p@k")
    plt.plot(x_axis, y_axis, label = 'SE'+str(index_top_5[t]+1))
    for a, b in zip(x_axis, y_axis):
        plt.annotate(str(b), xy=(a, b), fontsize = 6)
    plt.legend()
plt.show()

# Average nDCG@k plot for top 5 SEs
plt.figure()
for i, t in zip(index_top_5, range(5)):
    x_axis = k
    y_axis = []
    for j in k:
        y_axis.append(np.mean(nDCG_for_all(j, all_se[i], gt_d)))
    y_axis = np.round(y_axis, 5)
    plt.style.use('ggplot')
    plt.xlabel("k values")
    plt.ylabel("Average nDCG@k")
    plt.plot(x_axis, y_axis, label = 'SE'+str(index_top_5[t]+1))
    for a, b in zip(x_axis, y_axis):
        plt.annotate(str(b), xy=(a, b), fontsize = 6)
    plt.legend()
plt.show()
############################################################################################
############################################################################################