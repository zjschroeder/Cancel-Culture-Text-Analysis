# IN THE TWIST OF THE CENTURY, WE'RE TRANSITIONING BACK TO USING R FOR THE REST OF THE ANALYSES
# XOXO GOSSIP GIRL

# ------------------------------------------ CHUNK 1: Document Term Matrix Functions ------------------------------------------
import pandas as pd
import math
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("data/study1.csv", nrows=100)

# Count DTM
#Term Document Matrix
def count_dtm(column):
    vect = CountVectorizer()
    vects = vect.fit_transform(column)
    td = pd.DataFrame(vects.todense())
    td.columns = vect.get_feature_names_out()
    term_count_matrix = td.T
    term_count_matrix.columns = ['Doc ' + str(i) for i in range(1, len(column)+1)]
    # Term Frequency: Number of times word appears in document / total words in document
    words_in_doc = pd.DataFrame(term_count_matrix.sum()).T
    term_frequency = term_count_matrix.div(words_in_doc.iloc[0])
    # Inverse Document Frequency: log(Total documents in corpus / number of docs with term)

    return(term_count_matrix)


y = count_dtm(df.posttoken_bio)
docs_in_corpus = y.columns
n_term_uses = ((len(docs_in_corpus)) - (y[docs_in_corpus] == 0).astype(int).sum(axis=1))

inverse_doc_frequency = (len(docs_in_corpus)/n_term_uses).apply(math.lo)

# TF-IDF: TF * IDF
# Cosine Similarity Matrix: Term X Term cosine similarity matrix
### Minimum threshold of cos = .02
# Other metrics of interest
###
### times_w_used = term_document_matrix.sum(axis=1)
### n_docs_with_w = (len(cols_to_calc)) - (
    ### (term_document_matrix[cols_to_calc] == 0).astype(int).sum(axis=1))



# TODO: Figure out why this does/doesn't work on all dataframes

#Binary DTM
bin_vec = MultiLabelBinarizer()
mlb = bin_vec.fit(df['posttoken'])
term_document_matrix = pd.DataFrame(mlb.transform(df['posttoken']), columns=[mlb.classes_])

# ------------------------------------------ CHUNK 2: Create DTMs ------------------------------------------

# Study 1
study1_dtm_tweets = dtm(study1, "posttoken")
study1_dtm_tweets.to_csv("data/study_1_cancel_culture/study1_dtm_tweets.csv", index=False)
study1_dtm_bio = dtm(study1, "posttoken_bio")
study1_dtm_bio.to_csv("data/study_1_cancel_culture/study1_dtm_bio.csv", index=False)

study2_dtm_tweets = dtm(study2, "posttoken")
study2_dtm_tweets.to_csv("data/study_2_isoverparty/study2_dtm_tweets.csv", index=False)
study2_dtm_bio = dtm(study2, "posttoken_bio")
study2_dtm_bio.to_csv("data/study_2_isoverparty/study2_dtm_bio.csv", index=False)

study3_dtm_tweets = dtm(study3, "posttoken")
study3_dtm_tweets.to_csv("data/study_3_faculty/study3_dtm_tweets.csv", index=False)
study3_dtm_bio = dtm(study3, "posttoken_bio")
study3_dtm_bio.to_csv("data/study_3_faculty/study3_dtm_bio.csv", index=False)

study4_dtm_tweets = dtm(study4, "posttoken")
study4_dtm_tweets.to_csv("data/study_4_celebrity/study4_dtm_tweets.csv", index=False)
study4_dtm_bio = dtm(study4, "posttoken_bio")
study4_dtm_bio.to_csv("data/study_4_celebrity/study4_dtm_bio.csv", index=False)

# Document term matrices
study5_dtm_tweets = dtm(study5, "posttoken")
study5_dtm_tweets.to_csv("data/study_5_civilians/study5_dtm_tweets.csv", index=False)
study5_dtm_bio = dtm(study5, "posttoken_bio")
study5_dtm_bio.to_csv("data/study_5_civilians/study5_dtm_bio.csv", index=False)

is_over_dtm_tweets = dtm(is_over, "posttoken")
is_over_dtm_bio = dtm(is_over, "posttoken_bio")
is_over_dtm_tweets.to_csv("data/study_2_isoverparty/is_over_dtm_tweets.csv", index=False)
is_over_dtm_bio.to_csv("data/study_2_isoverparty/is_over_dtm_bio.csv", index=False)