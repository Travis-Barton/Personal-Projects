## embeddings loading

library(readr)
library(parallel)
library(textclean)



library(tidyr)
library(LilRhino)
library(forcats)
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
Num_Al_sep = function(vec){
  vec = unlist(strsplit(vec, "(?=[A-Za-z])(?<=[0-9])|(?=[0-9])(?<=[A-Za-z])", perl = TRUE))
  vec = paste(vec, collapse = " ")
  return(vec)
}
Pretreatment = function(title_vec, stem = TRUE, lower = TRUE, parallel = F){
  Num_Al_sep = function(vec){
    vec = unlist(strsplit(vec, "(?=[A-Za-z])(?<=[0-9])|(?=[0-9])(?<=[A-Za-z])", perl = TRUE))
    vec = paste(vec, collapse = " ")
    return(vec)
  }
  if(parallel == F){
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
  else{
    numcore = detectCores()
    titles = as.character(title_vec) %>%
      mclapply(gsub, pattern = "[^[:alnum:][:space:]]",replacement = "", mc.cores = numcore) %>%
      unlist() %>%
      mclapply(Num_Al_sep, mc.cores = numcore) %>%
      unlist() %>%
      mclapply(replace_number, mc.cores = numcore) %>%
      unlist()
    if(stem == TRUE){
      titles = titles %>%
        mclapply(stemDocument, mc.cores = numcore)
    }
    if(lower == TRUE){
      titles = titles %>% 
        mclapply(tolower, mc.cores = numcore)
    }
    return(titles)
  }
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
  glove = GlobalVectors$new(word_vectors_size = vector_size, 
                            vocabulary = vocab, 
                            x_max = 5, shuffle = T,
                            lambda = 1e-5)
  temp1 = glove$fit_transform(tcm, n_iter = iterations)
  return(as.data.frame(temp1 + t(glove$components)))
  #return(as.data.frame(glove$components))
}

##### New word converter
Vector_puller2 = function(words, emb_matrix, dimension){
  ret = colMeans(emb_matrix[words,], na.rm = TRUE)
  if(all(is.na(ret)) == T){
    return(rep(0, dimension))
  }
  return(ret)
}
#Make sure that the embeddings matrix is a data frame
Sentence_Vector2 = function(Sentences, emb_matrix, stopwords, dimension){
    words_list = stringi::stri_extract_all_words(Sentences, simplify = T)
    vecs = Vector_puller2(words_list, emb_matrix, dimension)
    return(t(vecs))
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

