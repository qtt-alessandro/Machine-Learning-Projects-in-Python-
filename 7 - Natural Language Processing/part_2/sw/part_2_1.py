import json
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from genre.hf_model import GENRE
import pandas as pd
import pickle

cwd = os.getcwd()
print("Current working directory: {0}".format(os.getcwd()))

"""
function to write pickle file
"""
def write_pickle(data, name):
    with open( str(name) +'.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""
function to read pickle file
"""
def read_pickle(name):
    with open( str(name)+'.pickle', 'rb') as handle:
        data = pickle.load(handle)
        return data

"""
function to read json file
"""
def read_json(name):
    with open(str(name)) as f:
        data = json.load(f)
    return data

"""
function to write jsonl file
"""
def write_jsonl(data,name):
    with open(str(name)+'.jsonl', 'w') as f:
        json.dump(data, f)

""""
This function is implemented to apply the model fom the library genre 
to predict some features according to the input sentence.
"""
def wikipedia_pages(sentence):
    sentence = [sentence]
    predictions = model_genre.sample(sentence)
    aas = predictions[0][0]['text']
    add = [p.split(']')[0] for p in aas.split('[') if ']' in p]
    return add

"""
This function map the function wikipedia_pages to the column 'wikipedia_pages'
"""
def adding_wikipedia_pages(df):
    tqdm.pandas()
    df['wikipedia_pages'] = df.input.progress_map(wikipedia_pages)
    return df

"""
This function uses the abstract_kilt data and the column wikipedia_pages in order to find 
the abstract of a wikipedia page given a input string.
If the lists have more than one element, for each of them return respectively the abstract and sort them 
in ascending order. 
At the end, all the abstracts are concatenated with a black empty space in that order.
Try/except has been added in order to have control of the cases where the wikipedia_pages has no match with abstract_kilt data.
In that cases, an empty string is added.
"""
def wikipedia_abstract(data, df):
    for i, row in tqdm(df.iterrows()):
        aux = []
        for elem in df.wikipedia_pages[i]:
            try:
                aux.append(data[elem[1:-1]])
                lst2 = sorted(aux, key=len, reverse = False)
                aa  = '  '.join(lst2)
                df["wikipedia_abstract"][i] = aa
            except:
                df["wikipedia_abstract"][i] = ' '
    return df

"""
In order to compute the embedding for each claim, a function that takes as input a string 
and returns an array of 768 elements with the probability log, is implemented.
The array is created accordingly to the model paraphrase-distilroberta-base-v1 from the SentenceTransformer library.
"""
def embeddings(sentence):
    model_emb.max_seq_length = 256 #extending the input sentence size to 256
    claim_embedding = model_emb.encode(sentence, show_progress_bar=False) #fitting the model for embedding computation on the input sentence
    claim_embedding_list = claim_embedding.tolist()
    return claim_embedding_list

"""
To apply the function the entire column of the dataset, the function map it is implemented. 
This function allows to reduce the number of loop in the code, reducing the complexity. 
"""
def adding_embeddings_abstract(df):
    tqdm.pandas()
    df['abstract_embedding'] = df.wikipedia_abstract.progress_map(embeddings)
    return df


model_genre   = GENRE.from_pretrained("./part_2/models/hf_e2e_entity_linking_wiki_abs").eval()
model_emb     = SentenceTransformer('paraphrase-distilroberta-base-v1')
abstract_kilt = read_json('part_2/input_data/abstract_kilt_knowledgesource.json')


#---------------------DEVIATION DATASET---------------------------------
emb_dev_dataframe = read_pickle("./part_1/output_data/emb_dev")
emb_dev_dataframe["wikipedia_pages"] = " "
emb_dev_dataframe = adding_wikipedia_pages(emb_dev_dataframe)
emb_dev_dataframe["wikipedia_abstract"] = " "
emb_dev_dataframe = wikipedia_abstract(abstract_kilt,emb_dev_dataframe)
emb_dev_dataframe["abstract_embedding"] = " "
emb_dev_dataframe = adding_embeddings_abstract(emb_dev_dataframe)
write_pickle(emb_dev_dataframe,"./part_2/output_data/emb_dev2")
emb_dev2 = emb_dev_dataframe.to_dict(orient='records')
write_jsonl(emb_dev2,"./part_2/output_data/emb_dev2")
#-----------------------------------------------------------------------


#---------------------TEST DATASET--------------------------------------
emb_test_dataframe = read_pickle("./part_1/output_data/emb_test")
emb_test_dataframe = emb_test_dataframe[:100]
emb_test_dataframe["wikipedia_pages"] = " "
emb_test_dataframe = adding_wikipedia_pages(emb_test_dataframe)
emb_test_dataframe["wikipedia_abstract"] = " "
emb_test_dataframe = wikipedia_abstract(abstract_kilt,emb_test_dataframe)
emb_test_dataframe["abstract_embedding"] = " "
emb_test_dataframe = adding_embeddings_abstract(emb_test_dataframe)
write_pickle(emb_test_dataframe,"./part_2/output_data/emb_test2")
emb_test2 = emb_dev_dataframe.to_dict(orient='records')
write_jsonl(emb_test2,"./part_2/output_data/emb_test2")
#-----------------------------------------------------------------------


#---------------------TRAIN DATASET-------------------------------------
emb_train_dataframe = read_pickle("./part_1/output_data/emb_train")
emb_train_dataframe = emb_train_dataframe[:100]
emb_train_dataframe["wikipedia_pages"] = " "
emb_train_dataframe = adding_wikipedia_pages(emb_train_dataframe)
emb_train_dataframe["wikipedia_abstract"] = " "
emb_train_dataframe = wikipedia_abstract(abstract_kilt,emb_train_dataframe)
emb_train_dataframe["abstract_embedding"] = " "
emb_train_dataframe = adding_embeddings_abstract(emb_train_dataframe)
write_pickle(emb_train_dataframe,"./part_2/output_data/emb_train2")
emb_train2 = emb_train_dataframe.to_dict(orient='records')
write_jsonl(emb_train2,"./part_2/output_data/emb_train2")
#-----------------------------------------------------------------------