## embeddings loading

library(readr)
library(tidyr)
library(e1071)
library(LilRhino)
library(forcats)
library(textclean)
library(tm)
library(text2vec)
library(fastmatch)
library(data.table)

## 0) glove embeddings loading
load_glove_embeddings = function(path, d){
  #'/Users/travisbarton/Downloads/glove.6B/glove.6B.50d.txt'
  col_names <- c("term", paste("d", 1:d, sep = ""))
  dat <- as.data.frame(read_delim(file = path,
                                  delim = " ",
                                  quote = "",
                                  col_names = col_names))
  rownames(dat) = dat$term
  dat = dat[,-1]
  return(dat)
}


## 0) Pretreatment
Pretreatment = function(title_vec, stem = TRUE, lower = TRUE){
  Num_Al_sep = function(vec){
    vec = unlist(strsplit(vec, "(?=[A-Za-z])(?<=[0-9])|(?=[0-9])(?<=[A-Za-z])", perl = TRUE))
    vec = paste(vec, collapse = " ")
    return(vec)
  }
  titles = as.character(title_vec) %>%
    lapply(gsub, pattern = "[^[:alnum:][:space:]]",replacement = "") %>%
    unlist() %>%
    lapply(Num_Al_sep) %>%
    unlist() %>%
    lapply(replace_number) %>%
    unlist()
  if(stem == TRUE){
    titles = titles %>%
      lapply(stemDocument)
  }
  if(lower == TRUE){
    titles = titles %>% 
      lapply(tolower)
  }
  return(titles)
}


## 1) Stopword maker
StopWordMaker = function(titles, cutoff = 20){
  test = unlist(lapply(as.vector(titles), strsplit, split = ' ', fixed = FALSE))
  stopwords = test %>%
    table() %>%
    sort(decreasing = TRUE) %>%
    head(cutoff) %>%
    names()
  return(stopwords)
}


## 2) Auto-creating word matrix
Embedding_Matrix = function(sent, vocab_min, stopwords, skip_gram, vector_size, iterations = 20){
  it   = itoken(unlist(sent), tolower, word_tokenizer)
  vocab = create_vocabulary(it, stopwords = stopwords)
  vocab <- prune_vocabulary(vocab, term_count_min = vocab_min)
  # Use our filtered vocabulary
  vectorizer <- vocab_vectorizer(vocab)
  # use window of 5 for context words
  tcm <- create_tcm(it, vectorizer, skip_grams_window = skip_gram)
  glove = GlobalVectors$new(word_vectors_size = vector_size, vocabulary = vocab, x_max = 5)
  glove$fit_transform(tcm, n_iter = iterations)
  return(glove$components)
}

##### New word converter
Vector_puller2 = function(words, emb_matrix, dimension){
  ret = colMeans(emb_matrix[words,])
  if(all(is.na(ret)) == T){
    return(rep(0, dimension))
  }
  return(ret)
}
#Make sure that the embeddings matrix is a data frame
Sentence_Vector2 = function(Sentences, emb_matrix, stopwords, dimension){
  words_list = stringi::stri_extract_all_words(Sentences)
  vecs = lapply(words_list, Vector_puller2, emb_matrix, dimension)
  return(vecs)
}


#### OUTDATED CODE

## 3) Sentence Converter
###### 3a) Word Puller
# Vector_puller = function(word, emb_matrix){
#   if(word %chin% rownames(emb_matrix)){
#     return(emb_matrix[word,])
#   }
#   else{
#     return(rep(0, ncol(emb_matrix)))
#   }
# }
##### 3b) Sentence Vector
# Sentence_Vector = function(sentence, emb_matrix, stopwords){
#   if(sentence == ""){
#     return(rep(0, 50))
#   }
#   words = strsplit(sentence, " ", fixed = TRUE)[[1]]
#   words = words[-which(words %chin% stopwords)]
#   if(length(words) == 0){
#     return(rep(0, 50))
#   }
#   vec = lapply(words, Vector_puller, emb_matrix)
#   if(is.null(vec) != T){
#     df = matrix(unlist(vec), ncol = 50, byrow = T)
#   }
#   else{
#     df = matrix(0, ncol = 50)
#   }
#   zeros = which(rowSums(df) == 0)
#   if(length(zeros) == 0)
#   {
#     df = apply(df, 2, mean)
#     return(df)
#   }
#   else if(length(zeros) == nrow(df)){
#     return(rep(0, 50))
#   }
#   else if(length(zeros) == nrow(df)-1){
#     return(df[-zeros,])
#   }
#   else{
#     df = apply(df[-zeros,], 2, mean)
#     return(df)
#   }
# }

