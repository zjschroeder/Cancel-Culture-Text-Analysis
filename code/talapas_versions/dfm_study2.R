
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

lib_words <- rio::import('data/partisan_terms.csv') %>% 
  filter(p == 2) %>% 
  select(terms) %>% 
  as_vector() %>% 
  paste(., collapse = "|")

conserv_words <- rio::import('data/partisan_terms.csv') %>% 
  filter(p == 1) %>% 
  select(terms) %>% 
  as_vector() %>% 
  paste(., collapse = "|")

replace_words <- function(string, words, replacement) {
  for (word in words) {
    string <- str_replace_all(string, word, replacement)
  }
  return(string)
}

csv_to_tfidf <- function(path_to_csv){
  # Read in csv
  df = read.csv(here::here(path_to_csv)) %>% 
    mutate(
      posttoken_bio = gsub("[[:punct:]]", "", posttoken_bio),
      posttoken = gsub("[[:punct:]]", "", posttoken)
      )
  
  tweets_binary_mat <- corpus(df, text_field = "posttoken") %>% 
    tokens() %>% 
    dfm(., verbose = FALSE, 
        remove_padding = TRUE) %>% 
    dfm_trim(., min_docfreq = 0.001, 
             docfreq_type = "prop") %>% 
    dfm_subset(., ntoken(.) > 0) %>% 
    dfm_weight(scheme = "boolean") %>% 
    as.matrix()
  
  # Partisan Aggregated Nodes
  postoken_bio_partisan <- lapply(df$posttoken_bio, 
                                  replace_words, 
                                  lib_words, 
                                  replacement = "liber")
  
  df$posttoken_bio_partisan <- lapply(postoken_bio_partisan, 
                                      replace_words, 
                                      conserv_words, 
                                      replacement = "conserv") %>% 
                                      unlist()
  
  
  # Term Frequencey-Inverse Document Frequency
  data_dfm <- corpus(df, text_field = "posttoken_bio") %>% 
    tokens() %>% 
    dfm(., 
        verbose = FALSE, 
        remove_padding = TRUE) %>% 
    dfm_trim(., 
             min_docfreq = 0.001, 
             docfreq_type = "prop") %>% 
    dfm_subset(., 
               ntoken(.) > 0)
  
  data_tfidf <- dfm_tfidf(data_dfm)
  
  data_binary_mat <- data_dfm %>% 
    dfm_weight(scheme = "boolean") %>% 
    as.matrix()
  
  # Cosine similarity matrix
  terms_cosine <- textstat_simil(data_tfidf, 
                                 margin = "features", 
                                 method = "cosine") %>% 
    as.matrix()
  
  #Graph of Terms and Connections
  terms_graph <- graph_from_adjacency_matrix(terms_cosine,
                                             mode = "undirected", 
                                             weighted = TRUE, 
                                             diag = FALSE)
  
  # Prepare for scatterplot
  scatter_TFIDF <- corpus(df, text_field = "posttoken_bio_partisan") %>% 
    tokens() %>% 
    dfm(., 
        verbose = FALSE, 
        remove_padding = TRUE) %>% 
    dfm_trim(., 
             min_docfreq = 0.001, 
             docfreq_type = "prop") %>% 
    dfm_subset(., 
               ntoken(.) > 0) %>% 
    dfm_tfidf()
  
  #compute partisan cosines
  scatter_COSliber <- as.data.frame(textstat_simil(scatter_TFIDF, 
                                                   scatter_TFIDF[, c("liber")], 
                                                   method = "cosine", 
                                                   margin = "features")) 
  
  scatter_COSconserv <- as.data.frame(textstat_simil(scatter_TFIDF, 
                                                     scatter_TFIDF[,   c("conserv")], 
                                                     method = "cosine", 
                                                     margin = "features")) 
  
  #Liberal cosines
  COSlib <- subset(scatter_COSliber, select = c("feature1", "cosine"))
  COSlib$libcos <- COSlib$cosine
  COSlib <- subset(COSlib, select = c("feature1", "libcos"))  
  COSlib <- COSlib[order(COSlib$libcos, decreasing = TRUE),]  
  COSlib <- COSlib[0:110,]  
  COSlib <- subset(COSlib, feature1!="conserv")  
  
  #Conservative cosines
  COScon <- subset(scatter_COSconserv, select = c("feature1", "cosine"))
  COScon$concos <- COScon$cosine
  COScon <- subset(COScon, select = c("feature1", "concos"))
  COScon <- COScon[order(COScon$concos, decreasing = TRUE),]
  COScon <- COScon[0:110,]
  COScon <- subset(COScon,feature1!='liber' )
  
  
  COSall <- full_join(COSlib, COScon)
  COSall["libcos"][is.na(COSall["libcos"])] <- 0
  COSall["concos"][is.na(COSall["concos"])] <- 0
  COSall$partisanlean <- (COSall$concos - COSall$libcos)
  COSall_collapsed <- COSall %>%
    mutate(party = case_when(partisanlean < 0 ~ 'Liberal',
                             partisanlean > 0 ~ 'Conservative'))
  COSall_collapsed <- COSall_collapsed[!grepl("behind",   COSall_collapsed$feature1),]
  COSall_collapsed <- COSall_collapsed[!grepl("er", COSall_collapsed$feature1),]
  
  data_scatter <- COSall_collapsed
  
  mid <- mean(data_scatter$partisanlean)
  
  output = list(
    df = df,
    data_tfidf = data_tfidf,
    data_binary_mat = data_binary_mat,
    tweets_binary_mat = tweets_binary_mat,
    tfidf_mat = as.matrix(data_tfidf), 
    cos_sim_mat = terms_cosine,
    cos_sim_fig = terms_graph,
    partisan_tfidf = scatter_TFIDF, 
    partisan_cos_scores = data_scatter
  )
}

# 
# study5 <- csv_to_tfidf("data/study_5_civilians/study5.csv")
# save(study5, file = "data/study5_dfm.RData")
# 
# study4 <- csv_to_tfidf("data/study_4_celebrity/study4.csv")
# save(study4, file = "data/study4_dfm.RData")
# 
# study3 <- csv_to_tfidf("data/study_3_faculty/study3.csv")
# save(study3, file = "data/study3_dfm.RData")

study2 <- csv_to_tfidf("data/study_2_isoverparty/study2.csv")
save(study2, file = "data/study2_dfm.RData")

# study1 <- csv_to_tfidf("data/study_1_cancel_culture/study1_cleaned.csv")
# save(study1, file = "data/study1_dfm.RData")
