import pandas as pd
from whoosh.analysis import RegexTokenizer
from tqdm import tqdm
import csv
import ast
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
    # the below set is created to avoid duplicates for shingles
    # it represents the all possible shingles for dataframe's specific column
    shingle_set = set()
    # at the end, each row of df will have shingles in a column named "shingles"
    new_column_for_df = []
    for i in tqdm(range(len(df))):
        # doc_len: number of words in the document
        doc_len = len(df[column][i])
        sub_column_for_df = []
        if doc_len >= w:
            for j in range(doc_len - w + 1):
                my_str = ""
                for step in range(w):
                    my_str = my_str + df[column][i][j + step]
                shingle_set.add(my_str)
                sub_column_for_df.append(my_str)
            new_column_for_df.append(sub_column_for_df)
        # the following condition is taken to make shingling possible also for the docs have less number of words
        # this statement is especially useful for song name which is short in general
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


# jaccard function calculates the jaccard similarity between 2 lists
def jaccard(list1, list2):
    intersect = set.intersection(set(list1), set(list2))
    union = set(list1).union(set(list2))
    result = len(intersect) / len(union)
    return result


# create_groups function is created in order to decrease computational time
# It is the fact that the desired jaccard similarity must be 1 between documents,
# groups are created to collect each document from the same length
# Because, if the documents having different length of shingles cannot acquire jaccard similarity = 1
def create_groups(df, shingle_ids_column, song_name_column):
    # max number of shingles in a doc is considered to create total number of lists
    number_of_list = max(df[shingle_ids_column].str.len())
    allTheLists = [[] for x in range(number_of_list)]
    allTheLists_Songs = [[] for x in range(number_of_list)]
    # each doc having the same number of shingles are corresponded to the same list
    for i in range(len(df)):
        length = len(df[shingle_ids_column][i])
        allTheLists[length-1].append(df[shingle_ids_column][i])
        allTheLists_Songs[length-1].append(df[song_name_column][i])
    # in order to not get error in the upcoming functions, empty lists are removed
    empty_lists = []
    for i in range(len(allTheLists)):
        if len(allTheLists[i]) == 0:
            empty_lists.append(i)
    if len(empty_lists) != 0:
        for i in empty_lists:
            del allTheLists[i]
            del allTheLists_Songs[i]
    return allTheLists, allTheLists_Songs


# function in the below is created to return the number of couples of docs that have jaccard similarity score 1
def number_docs_jaccard_one(my_lists, my_lists_songs):
    result_sum = 0
    song_couples = []
    # it goes list by list, where lists have the same number of shingles
    for one_list in tqdm(range(len(my_lists))):
        for i in range(len(my_lists[one_list])):
            for j in range(len(my_lists[one_list])):
                # to remove duplicates such as j(1,2) = 1 & j(2,1) = 1, the below constraint is added
                if i < j:
                    if jaccard(my_lists[one_list][i], my_lists[one_list][j]) == 1:
                        result_sum += 1
                        song_couples.append([my_lists_songs[one_list][i], my_lists_songs[one_list][j]])
    return result_sum, song_couples

# The interest is only "song" column
col_list = ["song"]
df = pd.read_csv("./part_2/part_2_1/dataset/250K_lyrics_from_MetroLyrics.csv", usecols=col_list)

# nan values are eliminated and reindexing is made
df = df.dropna().reset_index(drop=True)

# remove punctuation and lower_case_filtering
# the new column, "song_", is made to see new version for song name
song_ = []
for i in tqdm(range(len(df["song"]))):
    song_.append(remove_punc_and_lower(df["song"][i]))
df["song_"] = song_

my_shingle_set = shingling(3, df, "song_")
my_shingle_dic = create_shingle_dic(my_shingle_set)
new_column_with_shingle_id(df, my_shingle_dic)
my_lists_ids, my_lists_songs = create_groups(df, "shingle_ids", "song")
number, dup_songs = number_docs_jaccard_one(my_lists_ids, my_lists_songs)
print(number)

with open('./part_2/part_2_2/dups_output.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(["song_name_1", "song_name_2"])
    for i in dup_songs:
        tsv_writer.writerow(i)



mask = (df['shingle_ids'].str.len() == 1)
data = df.loc[mask]
print(len(data))
print(len(df))
duplicates = 0
visited = []
cnt = 0
with open('./part_2/part_2_1/dataset/duplicates_method_output_test.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['song_id', 'duplicates_list'])
    for value in tqdm(data['shingle_ids'].str[0]):
        idx_duplicates_list = data.index[data['shingle_ids'].str[0] == value].tolist()
        tsv_writer.writerow([data.iloc[cnt].song, idx_duplicates_list])
        cnt +=1


duplicates_dataset = pd.read_csv('./part_2/part_2_1/dataset/duplicates_2_method_output.tsv', sep='\t', header=0)
#Returns the exact number of duplicates for only one shingle case
dupl = 0
for element in tqdm(duplicates_dataset.duplicates_list):
    if (len(ast.literal_eval(element))) >=2:
        dupl += 1
print(dupl)
print(len(data))
dupl*100/(len(data))
dupl*100/(len(df))