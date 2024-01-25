import pandas as pd
from nltk.corpus import stopwords
import nltk
import re
import csv
import contractions
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

#########################################
# DATA IMPORT
# df = pd.read_csv("data/cc_mentions.csv")
# df = df[['tweet_id', 'text']]
# df.to_csv('data/cc_id_text_raw.csv')
df = pd.read_csv("data/cc_id_text_raw.csv",
                 dtype={'tweet_id': 'str',
                        'text': 'str'})
df['text'] = df['text'].astype(str) # For whatever reason, you need to forcefully coerce

#########################################
# PRE-TOKENIZATION STRING CLEANING

stop_words = set(stopwords.words('english'))

def strg_list_to_list(strg_list):
    return strg_list.strip("[]").replace("'", "").replace('"', "").replace(",", "").split()

def remove_retweet_label(text):
    return re.sub('RT @[\w_]+:', '', text)

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

df['pretoken'] = df['text'].apply(pretokenization_cleaning)

##############################################
# TOKENIZATION

def tokenize(text):
    tknzr = TweetTokenizer(reduce_len=True)
    return tknzr.tokenize(text)

df['token'] = df['text'].apply(tokenize)

##############################################
# NORMALIZING (STEMMER)

def stemming(unkn_input):
    porter = nltk.PorterStemmer()
    if (isinstance(unkn_input, list)):
        list_input = unkn_input
    if (isinstance(unkn_input, str)):
        list_input = strg_list_to_list(unkn_input)
    list_stemmed = []
    for word in list_input:
        word = porter.stem(word)
        list_stemmed.append(word)
        # Two options for what to return:
    # return " ".join(list_stemmed) #string
    return list_stemmed  # list

df['stemmed'] = df['token'].apply(stemming)

# NOTE:
# You can either stem (reduce to word stems) or lemmatize when cleaning words.
# I prefer lemmatization as it seems a bit more accurate.

##############################################
# NORMALIZING (LEMMATIZER)

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


# lemmatize requires list input
def lemmatize(unkn_input):
    if (isinstance(unkn_input, list)):
        list_input = unkn_input
    if (isinstance(unkn_input, str)):
        list_input = strg_list_to_list(unkn_input)
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

# LEMMATIZING
df['lemmatized'] = df['token'].apply(lemmatize)

###########################################
# POST-TOKENIZATION TASKS:
# the following post-tokenization receives list as input parameter
# and returns list as output

def remove_punc(list_token):
    # print(list_token)
    def process(strg_token):
        strg_numb = '''0123456789'''
        strg_3dots = '...'
        strg_2dots = ".."
        strg_punc = '''!()+-[]{}|;:'"\,<>./?@#$£%^&*_~“”…‘’'''
        strg_output = ''
        # for idx, char in enumerate(strg_token):
        # print(item)
        if (len(strg_token) == 0):  # empty char
            strg_output += ''
        else:
            if (all(char in strg_numb for char in strg_token) or
                    strg_token[0] in strg_numb):  # if char is a number
                strg_output += ''
            else:
                if (len(strg_token) == 1 and strg_token in strg_punc):  # if char is a single punc
                    strg_output += ''
                else:
                    if (strg_token[0] == '#'):  # if char is hashtag
                        strg_output += strg_token.lower()
                    elif (strg_token == strg_3dots or strg_token == strg_2dots):
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
    if (isinstance(unkn_input, list)):
        list_output = unkn_input
    if (isinstance(unkn_input, str)):
        list_output = strg_list_to_list(unkn_input)
    list_output = remove_punc(list_output)
    list_output = remove_empty_item(list_output)
    # list_output=lowercase_alpha(list_output)

    return (list_output)

# Removes empty tokens, single punctuation, etc.
df['posttoken'] = df['lemmatized'].apply(posttokenization_cleaning)

#################################
# EXPORT CLEANED DATA AS CSV

df.to_csv('data/pre_processed_tweets.csv')

