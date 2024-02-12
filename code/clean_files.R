library(quanteda)
library(tidyverse)
library(quanteda.textstats)
library(igraph)
library(ggplot2)
library(ggrepel)
library(pheatmap)
library(RColorBrewer)
library(viridis)
library(stringr)
library(rlang)
library(gdata)
here::here()

clean_rdata <- function(path, study_list){
  load(path) # Load in .RData item
  
  list2env(get(study_list), envir = globalenv()) # Split into variables
  
  # Tweet DFM
  tweets_dfm <- corpus(df, text_field = "posttoken") %>% 
    tokens() %>% 
    dfm(., verbose = FALSE, 
        remove_padding = TRUE) %>% 
    dfm_trim(., min_docfreq = 0.001, 
             docfreq_type = "prop") %>% 
    dfm_subset(., ntoken(.) > 0)
  # Tweet Count Matrix
  tweets_count_matrix <- tweets_dfm %>% 
    as.matrix()
  
  tweets_tfidf <- dfm_tfidf(tweets_dfm)
  
  # Bio DFM
  bio_dfm <- corpus(df, text_field = "posttoken_bio") %>% 
    tokens() %>% 
    dfm(., verbose = FALSE, 
        remove_padding = TRUE) %>% 
    dfm_trim(., min_docfreq = 0.001, 
             docfreq_type = "prop") %>% 
    dfm_subset(., ntoken(.) > 0)
  # Bio Count Matrix
  bio_count_matrix <- bio_dfm %>% as.matrix() 
  
  # Rename variables
  gdata::mv(from = "data_binary_mat", to = "bio_binary_matrix")
  gdata::mv(from = "tweets_binary_mat", to = "tweets_binary_matrix")
  gdata::mv(from = "tfidf_mat", to = "bio_tfidf_matrix")
  gdata::mv(from = "cos_sim_mat", to = "bio_cosine_similarity_matrix")
  gdata::mv(from = "cos_sim_fig", to = "bio_cosine_similarity_figure")
  gdata::mv(from = "partisan_tfidf", to = "bio_partisan_terms_tfidf")
  gdata::mv(from = "partisan_cos_scores", to = "bio_partisan_cosine_scores")
  gdata::mv(from = "data_tfidf", to = "bio_tfidf")
  save(df, 
       # For tweets and bios: document frequency matrix, count and binary doc x token, term frequency-inverse document frequency
       tweets_dfm, tweets_count_matrix, tweets_binary_matrix, tweets_tfidf,
       bio_dfm, bio_count_matrix, bio_binary_matrix, bio_tfidf,
       # Cosine similarity and partisan terms for political categorization
       bio_cosine_similarity_matrix, bio_cosine_similarity_figure,
       bio_partisan_terms_tfidf, bio_partisan_cosine_scores,
       file = path
       )
  rm(list = setdiff(ls(), lsf.str()))
  gc()
}

clean_rdata("twitter_cancel_culture/data/study5_dfm.RData", "study5")
clean_rdata("twitter_cancel_culture/data/study4_dfm.RData", "study4")
clean_rdata("twitter_cancel_culture/data/study3_dfm.RData", "study3")
clean_rdata("twitter_cancel_culture/data/study2_dfm.RData", "study2")
clean_rdata("twitter_cancel_culture/data/study1_dfm.RData", "study1")

