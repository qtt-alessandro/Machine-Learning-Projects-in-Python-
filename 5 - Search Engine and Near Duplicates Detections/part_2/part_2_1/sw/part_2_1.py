import pandas as pd
from whoosh.analysis import RegexTokenizer
from tqdm import tqdm
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import matplotlib.pyplot as plt
import numpy as np
import csv

# This function removes punctuation and apply lower case
def remove_punc_and_lower(sentence):
    tokenizer = RegexTokenizer()
    my_list = []
    for token in tokenizer(sentence):
        my_list.append(token.text.lower())
    return my_list


# shingling function takes the following parameters:
# w: length of the shingle, df: dataframe(df), column: column to be applied shingling
def shingling(w, df, column):
    shingle_set = set()
    new_column_for_df = []
    for i in tqdm(range(len(df))):
        # doc_len: number of words in the document
        doc_len = len(df[column][i])
        sub_column_for_df = []
        if doc_len >= w:
            for j in range(doc_len - w + 1):
                my_str = ""
                for step in range(w):
                    my_str = my_str + df[column][i][j+step]
                shingle_set.add(my_str)
                sub_column_for_df.append(my_str)
            new_column_for_df.append(sub_column_for_df)
        else:
            my_str = ''.join(df[column][i])
            shingle_set.add(my_str)
            sub_column_for_df.append(my_str)
            new_column_for_df.append(sub_column_for_df)
    df["shingles"] = new_column_for_df
    return shingle_set

# shingle_dic: keys are the specific shingle(string), values are the shingle_id, so each key has only one value
def create_shingle_dic(my_shingle_set):
    shingle_dic = {}
    for i, j in zip(my_shingle_set, range(len(my_shingle_set))):
        shingle_dic[i] = j
    return shingle_dic

# create a new column to df and match each shingle(string) with its shingle_id
def new_column_with_shingle_id(df, my_shingle_dic):
    new_column_for_shingle_id = []
    for i in range(len(df)):
        shingle_id_list = []
        for j in range(len(df["shingles"][i])):
            shingle_id_list.append(my_shingle_dic[df["shingles"][i][j]])
        new_column_for_shingle_id.append(list(set(shingle_id_list)))
    df["shingle_ids"] = new_column_for_shingle_id

def probability_distribution_plot(r,b,jaccard_threshold, probability_value):
    fig, ax = plt.subplots(figsize=(10, 8))
    # Set axis ranges; by default this will put major ticks every 25.
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)

    j = np.linspace(0,1, 1000)
    p= 1-(1-j**r)**b
    plt.plot(j, p, color="red", linewidth=3,label='First Line')
    plt.axvline(x=jaccard_threshold,color='black', linestyle='--',linewidth=3,label='S Line')
    plt.scatter(jaccard_threshold,probability_value, color = "blue", linewidth= 25)


    plt.title('Probability distribution of near duplicates according to r=13, b=10 and the Jaccard Similarity threshold \n')
    plt.xlabel("Jaccard Similarity")
    plt.ylabel("Near Duplicates Probability")


    # Change major ticks to show every 20.
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))

    # Change minor ticks to show every 5. (20/4 = 5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))

    # Turn grid on for both major and minor ticks and style minor slightly
    # differently.
    ax.grid(which='major', color='#CCCCCC', linestyle='--')
    ax.grid(which='minor', color='#CCCCCC', linestyle=':')
    plt.savefig('S-curve.png')
    plt.show()


# csv to dataframe conversion is made and song and lyrics columns are formed
col_list = ["song", "lyrics"]
df = pd.read_csv("./part_2/part_2_1/dataset/250K_lyrics_from_MetroLyrics.csv", usecols=col_list)
# nan values are eliminated and reindexing is made
df = df.dropna().reset_index(drop=True)

#remove punctuation and lower_case_filtering is applied to lyrics column
for i in tqdm(range(len(df["lyrics"]))):
    df["lyrics"][i] = remove_punc_and_lower(df["lyrics"][i])

my_shingle_set = shingling(3, df, "lyrics")
my_shingle_dic = create_shingle_dic(my_shingle_set)
new_column_with_shingle_id(df, my_shingle_dic)

# write df to tsv, only extract shingle_ids column and docid(index of the dataframe)
# Note that this operation is made because this is a group work and computational/waiting time is wanted to decreased
# by tsv file creation

#next 3 line of code are usefull to look for rows with less than 2 shingles than can occur errors in java tools
mask = (df['shingle_ids'].str.len() <= 2)
data = df.loc[mask].index.tolist()
df = df.drop(data)

df[["song","shingle_ids"]].to_csv("./part_2/part_2_1/dataset/250K_lyrics_from_MetroLyrics.csv", sep="\t", index = False)
#####################################################################
#how to get the parameter
with open('./part_2/part_2_1/dataset/output.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['probability', 'r', 'b'])
    r = np.arange(start=1, stop=301, step=1)
    b = np.arange(start=1, stop=301, step=1)
    for i in r:
        for j in b:
            if i*j<300:
                p= 1-(1-.95**i)**j
                if p > 0.97 and p < 1:
                    tsv_writer.writerow([p,i,j])
    print("output.tsv file generated")




probability_distribution_plot(r=13,b=10,jaccard_threshold=0.95, probability_value=0.9992548599156004)


addr_out = "./part_2/part_2_1/dataset/dataset_LSH__13_10_near_duplicates_output.tsv"
near_duplicates = pd.read_csv(addr_out, sep='\t', header=0)
near_duplicates_numbers = len(near_duplicates)
songs_numbers = len(df)
duplicates_percentage = (near_duplicates_numbers*100)/songs_numbers
print("Number of near duplicates founded according to the parameters:",near_duplicates_numbers)
print("Duplicate songs percentage: "+str(round(duplicates_percentage,2)) + "%")
