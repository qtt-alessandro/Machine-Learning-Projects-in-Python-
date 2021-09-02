import json
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd
import pickle

cwd = os.getcwd()
print("Current working directory: {0}".format(os.getcwd()))
"""
function to write pickle file
"""
def write_pickle(data, name):
    with open(str(name) + '.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""
function to read pickle file
"""
def read_pickle(name):
    with open(str(name) + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
        return data

"""
function to write jsonl file
"""
def write_jsonl(data, name):
    with open(str(name) + '.jsonl', 'w') as f:
        json.dump(data, f)

"""
function to convert jsonl as a list of dictionaries
"""
def jsonl_to_json(name):
    with open(str(name) + '.jsonl') as f:
        data = [json.loads(line) for line in tqdm(f)]
        print("data loading completed")
        return data

"""
function to setup, accordingly to the requests, the format of the output fields
"""
def setup_output_dictionary(df):
    for index, row in tqdm(df.iterrows()):
        #extr takes the first dictionary of the list of dictionaries
        #and select the label REFUTES or SUPPORT
        #in order to create an auxiliary dictionary containing only one label and one field
        extr = df.output[index][0]['answer']
        #erasing column at the given index
        df.output[index] = " "
        #substitute the field with the correct variable
        df.output[index] = [{"answer": extr}]
    return df
"""
In order to compute the embedding for each claim, a function that takes as input a string
and returns an array of 768 elements with the probability log is created.
The array is created accordingly to the model paraphrase-distilroberta-base-v1 from the SentenceTransformer library.
"""
def embeddings(sentence):
    claim_embedding = model_emb.encode(sentence, show_progress_bar=False)
    claim_embedding_list = claim_embedding.tolist()
    return claim_embedding_list
"""
To apply the function the entire column of the dataset, the function map it is implemented.
This function allows to reduce the number of loop in the code, reducing the complexity.
"""
def adding_embeddings(df):
    tqdm.pandas()
    #mapping the column, after creating, the embedding column.
    #progress_map is a function from tqdm library to show the progress bar of the execution.
    df['claim_embedding'] = df.input.progress_map(embeddings)
    return df

#model used for the embedding computation
model_emb = SentenceTransformer('paraphrase-distilroberta-base-v1')

"""
converting jsoln into a list of json dictionaries
"""

#---------------------------------------------------------
dev_dict   = jsonl_to_json("./part_1/input_data/dev_in")
test_dict  = jsonl_to_json("./part_1/input_data/test_in")
train_dict = jsonl_to_json("./part_1/input_data/train_in")
#---------------------------------------------------------

"""
converting the list of dictionaries into a Dataframe
"""
#---------------------------------------------------------
dev_dataframe  = pd.DataFrame(dev_dict)
test_dataframe = pd.DataFrame(test_dict)
train_dataframe =pd.DataFrame(train_dict)
#---------------------------------------------------------


"""
applying all the function seen above to the the deviation dataset,
saving a pickle file to store a backup copy of the file and, at the end, convert it to jsonl.
"""
#---------------------DEV DATASET------------------------
dev_dataframe = setup_output_dictionary(dev_dataframe)
dev_dataframe["claim_embedding"] = " "
dev_dataframe = adding_embeddings(dev_dataframe)
write_pickle(dev_dataframe,"./part_1/output_data/emb_dev")
dev_dataframe = dev_dataframe.to_dict(orient='records')
write_jsonl(dev_dataframe,"./part_1/output_data/emb_dev")
#---------------------------------------------------------


"""
applying all the function seen above to the the test dataset,
saving a pickle file to store a backup copy of the file and, at the end, convert it to jsonl.
"""
#---------------------TEST DATASET------------------------
test_dataframe = adding_embeddings(test_dataframe)
write_pickle(test_dataframe,"./part_1/output_data/emb_test")
test_dataframe = test_dataframe.to_dict(orient='records')
write_jsonl(test_dataframe,"./part_1/output_data/emb_test")
#---------------------------------------------------------

"""
applying all the function seen above to the the train dataset,
saving a pickle file to store a backup copy of the file and, at the end, convert it to jsonl.
"""
#---------------------TRAIN DATASET-----------------------
train_dataframe = setup_output_dictionary(train_dataframe)
train_dataframe["claim_embedding"] = " "
train_dataframe = adding_embeddings(train_dataframe)
write_pickle(train_dataframe,"./part_1/output_data/emb_train")
train_dataframe = train_dataframe.to_dict(orient='records')
write_jsonl(train_dataframe,"./part_1/output_data/emb_train")
#---------------------------------------------------------
