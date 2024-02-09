# ------------------------------------------------- CHUNK 1: IMPORT -------------------------------------------------
import pandas as pd
from nltk.corpus import stopwords
import nltk
import re
from datetime import datetime
import contractions
import os
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')

# ------------------------------------------ CHUNK 2: FUNCTION TO MERGE JSONS ------------------------------------------
def merge_json_files(folder_path):
    # Get a list of all JSON files in the specified folder
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    # Initialize an empty list to store dataframes and original file names
    dfs = []

    # Iterate through each JSON file and append dataframe and file name to the list
    for file in json_files:
        file_path = os.path.join(folder_path, file)

        # Read JSON file into a dataframe
        json_df = pd.read_json(file_path, orient='records')

        # Add a new column for the original file name without the ".json" suffix
        json_df['original_filename'] = os.path.splitext(file)[0]

        # Append dataframe and file name to the list
        dfs.append(json_df)

    # Concatenate all dataframes in the list
    merged_df = pd.concat(dfs, axis=0, ignore_index=True, sort=False)

    return merged_df


# merge_json_path = '/Users/zjschroeder/PycharmProjects/Cancel-Culture-Text-Analysis/data/combined_json'
# merged_jsons = merge_json_files(merge_json_path)
# ------------------------------------------ CHUNK 3: PRE-TOKEN DATA CLEANING ------------------------------------------

stop_words = set(stopwords.words('english'))


def strg_list_to_list(strg_list):
    return strg_list.strip("[]").replace("'", "").replace('"', "").replace(",", "").split()


def remove_retweet_label(text):
    return re.sub('RT @[A-Za-z0-9_-]+:', '', text)


def remove_video_label(text):
    return re.sub('VIDEO:', '', text)


def remove_hyperlink(text):
    return re.sub(r'http\S+', '', text)  # r=raw \S=string


def remove_twitterhandle(text):
    return re.sub('@[A-Za-z0-9_]+(:)?', '', text)


def remove_escape_sequence(text):
    return re.sub(r'\n', '', text)


def remove_extra_spaces(text):
    return re.sub(r"\s+", " ", text)


def remove_contractions(text):
    return contractions.fix(text)


def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stop_words])


def pretokenization_cleaning(text):
    text = remove_retweet_label(text)
    text = remove_video_label(text)
    text = remove_hyperlink(text)
    text = remove_twitterhandle(text)
    text = remove_escape_sequence(text)
    text = remove_extra_spaces(text)
    text = remove_contractions(text)
    text = remove_stopwords(text)
    return text


# ------------------------------------------ CHUNK 4: TOKENIZATION ------------------------------------------

def tokenize(text):
    tknzr = TweetTokenizer(reduce_len=True)
    return tknzr.tokenize(text)


# ------------------------------------------ CHUNK 5: STEMMER FUNCTION ------------------------------------------

# def stemming(unkn_input):
#     porter = nltk.PorterStemmer()
#     if (isinstance(unkn_input, list)):
#         stem_list_input = unkn_input
#     if (isinstance(unkn_input, str)):
#        stem_list_input = strg_list_to_list(unkn_input)
#    list_stemmed = []
#    for word in stem_list_input:
#        word = porter.stem(word)
#        list_stemmed.append(word)
#        # Two options for what to return:
#    # return " ".join(list_stemmed) #string
#    return list_stemmed  # list

# ------------------------------------------ CHUNK 6: LEMMATIZER ------------------------------------------

lemmatizer = WordNetLemmatizer()


def nltk_pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize(unkn_input):
    if isinstance(unkn_input, list):
        list_input = unkn_input
    elif isinstance(unkn_input, str):
        list_input = strg_list_to_list(unkn_input)
    else:
        list_input = strg_list_to_list("THIS IS AN ERROR")
    list_sentence = [item.lower() for item in list_input]
    nltk_tagged = nltk.pos_tag(list_sentence)
    wordnet_tagged = map(lambda x: (x[0], nltk_pos_tagger(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        # " ".join(lemmatized_sentence)
    return lemmatized_sentence


# ------------------------------------------ CHUNK 7: POST-TOKEN CLEANING  ------------------------------------------
def remove_punc(list_token):
    # print(list_token)
    def process(strg_token):
        strg_numb = '''0123456789'''
        strg_3dots = '...'
        strg_2dots = ".."
        strg_punc = '''!()+-[]{}|;:'"\\,<>./?@#$£%^&*_~“”…‘’'''
        strg_output = ''
        # for idx, char in enumerate(strg_token):
        # print(item)
        if len(strg_token) == 0:  # empty char
            strg_output += ''
        else:
            if (all(char in strg_numb for char in strg_token) or
                    strg_token[0] in strg_numb):  # if char is a number
                strg_output += ''
            else:
                if len(strg_token) == 1 and strg_token in strg_punc:  # if char is a single punc
                    strg_output += ''
                else:
                    if strg_token[0] == '#':  # if char is hashtag
                        strg_output += strg_token.lower()
                    elif strg_token == strg_3dots or strg_token == strg_2dots:
                        strg_output += ''
                    else:  # other than above, char could be part of word,
                        # e.g key-in
                        strg_output += strg_token
        return strg_output

    list_output = [process(token) for token in list_token]
    return list_output


def remove_empty_item(list_item):
    token = [token for token in list_item if len(token) > 0]
    return token


def lowercase_alpha(list_token):
    return [token.lower() if (token.isalpha() or token[0] == '#') else token for token in list_token]


def posttokenization_cleaning(unkn_input):
    list_output = []
    if isinstance(unkn_input, list):
        list_output = unkn_input
    if isinstance(unkn_input, str):
        list_output = strg_list_to_list(unkn_input)
    list_output = remove_punc(list_output)
    list_output = remove_empty_item(list_output)
    # list_output=lowercase_alpha(list_output)

    return list_output


# ------------------------------------------ CHUNK 8: Full Function ------------------------------------------
# Function to Clean Tweets
def clean_tweets(df):
    df['user_description'] = df['text'].astype(str)
    df['pretoken'] = df['text'].apply(pretokenization_cleaning)
    df['token'] = df['pretoken'].apply(tokenize)
    df['lemmatized'] = df['token'].apply(lemmatize)
    df['posttoken'] = df['lemmatized'].apply(posttokenization_cleaning)
    return df


# Function to Clean Bios
def clean_bios(df):
    df['user_description'] = df['user_description'].astype(str)
    df['pretoken_bio'] = df['user_description'].apply(pretokenization_cleaning)
    df['token_bio'] = df['pretoken_bio'].apply(tokenize)
    df['lemmatized_bio'] = df['token_bio'].apply(lemmatize)
    df['posttoken_bio'] = df['lemmatized_bio'].apply(posttokenization_cleaning)
    return df


# Function to process each file
def process_file(file_path):
    # Determine file type based on extension
    file_extension = os.path.splitext(file_path)[-1].lower()

    if file_extension == '.json':
        df = pd.read_json(file_path,
                          dtype={'tweet_id': 'str',
                                 'text': 'str',
                                 'user_description': 'str'})
    elif file_extension == '.csv':
        df = pd.read_csv(file_path,
                         dtype={'tweet_id': 'str',
                                'text': 'str',
                                'user_description': 'str'})
    else:
        # Skip files with unsupported extensions
        return None

    # Apply the clean_tweets function
    df = df[df['lang'] == 'en']
    df = clean_tweets(df)
    df = clean_bios(df)
    # TODO: Fix the errors in string length for bios
    # Add a new column for the original filename without extension
    filename = os.path.splitext(os.path.basename(file_path))[0]
    df['original_filename'] = filename

    return df


# ------------------------------------------ CHUNK 9: Clean Datasets  ------------------------------------------

# Study 1
study1 = process_file("data/study_1_cancel_culture/raw_data/cc_full.csv")
study1.to_csv("data/study_1_cancel_culture/study1_cleaned.csv", index=False)
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Study 1 done at =", current_time)

# Study 2
isover1 = process_file("data/study_2_isoverparty/isover1.csv")
isover2 = process_file("data/study_2_isoverparty/isover2.csv")
isoverparty = process_file("data/study_2_isoverparty/isoverparty.csv")
# is_over = process_file("data/study_2_isoverparty/is_over.csv")
# TODO: Identify error in is_over dataframe cleaning
study2 = pd.concat([isover1, isover2, isoverparty], ignore_index=True)
study2.to_csv("data/study_2_isoverparty/study2.csv", index=False)
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Study 2 done at =", current_time)

# Study 3
faculty_2 = process_file("data/study_3_faculty/faculty_2.csv")
faculty_3 = process_file("data/study_3_faculty/faculty_3.csv")
faculty_query2_1 = process_file("data/study_3_faculty/faculty_query2_1.csv")
faculty = process_file("data/study_3_faculty/faculty.csv")
study3 = pd.concat([faculty_2, faculty_3, faculty_query2_1, faculty], ignore_index=True)
study3.to_csv("data/study_3_faculty/study3.csv")
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Study 3 done at =", current_time)

# Study 4
aaron_rodgers = process_file('data/study_4_celebrity/aaron_rodgers.csv')
armie_hammer = process_file('data/study_4_celebrity/armie_hammer.csv')
# dave_chappelle = process_file('data/study_4_celebrity/dave_chappelle.csv')
# TODO: Fix error with string length in dave_chappelle
ellen = process_file('data/study_4_celebrity/ellen.csv')
james_charles = process_file('data/study_4_celebrity/james_charles.csv')
kanye = process_file('data/study_4_celebrity/kanye.csv')
r_kelly = process_file('data/study_4_celebrity/r_kelly.csv')
shane_dawson = process_file('data/study_4_celebrity/shane_dawson.csv')
# travis_scott = process_file('data/study_4_celebrity/travis_scott.csv')
# TODO: Fix error in string length travis scott
dojacat = process_file('data/study_4_celebrity/dojacat.json')
kanyewest = process_file('data/study_4_celebrity/kanyewest.json')
lindsayellis = process_file('data/study_4_celebrity/lindsayellis.json')
rkelly = process_file('data/study_4_celebrity/rkelly.json')
will_smith_full = process_file('data/study_4_celebrity/will_smith_full.json')
study4 = pd.concat([aaron_rodgers, armie_hammer, ellen,
                    james_charles, kanye, r_kelly,
                    shane_dawson, dojacat, kanyewest,
                    lindsayellis, rkelly, will_smith_full],
                   ignore_index=True)
study4.to_csv("data/study_4_celebrity/study4.csv")
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Study 4 done at =", current_time)

# Study 5
AaronMSchlossberg = process_file('data/study_5_civilians/AaronMSchlossberg.csv')
bbqbecky = process_file('data/study_5_civilians/bbqbecky.csv')
PoolPatrolPaula = process_file('data/study_5_civilians/PoolPatrolPaula.csv')
RhondaPolon = process_file('data/study_5_civilians/RhondaPolon.csv')
# bbqbeckyj = process_file('data/study_5_civilians/bbqbecky.json')
# justinesacco = process_file('data/study_5_civilians/justinesacco.json')
# permitpatty = process_file('data/study_5_civilians/permitpatty.json')
# TODO: All json files in study 5 encounter errors with string length
study5 = pd.concat([AaronMSchlossberg, bbqbecky, PoolPatrolPaula, RhondaPolon])
study5.to_csv("data/study_5_civilians/study5.csv")
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Study 5 done at =", current_time)

# -------------------------------------- CHUNK 10: TERM DOCUMENT MATRIX --------------------------------------

test_df = pd.read_csv("data/study1.csv", nrows=100)
# Count Vectorizer
vect = CountVectorizer()
vects = vect.fit_transform(test_df.posttoken)

# Select the first five rows from the data set
td = pd.DataFrame(vects.todense())
td.columns = vect.get_feature_names_out()
term_document_matrix = td.T
term_document_matrix.columns = ['CC_tweet ' + str(i) for i in range(1, len(term_document_matrix.axes[1]) + 1)]
term_document_matrix['total_count'] = term_document_matrix.sum(axis=1)

# Top 25 words
term_document_matrix = term_document_matrix.sort_values(by='total_count', ascending=False)[:25]

# Print the first 10 rows
print(term_document_matrix.drop(columns=['total_count']).head(10))

# Quick Visualization
term_document_matrix['total_count'].plot.bar()

# MEANING EXTRACTION METHOD DATA PREP

# Set word threshold: Filter tweets with <5 words

# Create counts of words that appear in < 10% of Tweets

# Create the matrix that is all tweets by all words binary 0/1

# PCA
