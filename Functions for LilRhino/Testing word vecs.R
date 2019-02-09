## embeddings loading

library(readr)

load_glove_embeddings = function(path, d){
  #'/Users/travisbarton/Downloads/glove.6B/glove.6B.50d.txt'
  col_names <- c("term", paste("d", 1:d, sep = ""))
  dat <- as.data.frame(read_delim(file = path,
                                  delim = " ",
                                  quote = "",
                                  col_names = col_names))
  
  rownames(dat) = dat$term
  dat = dat[,-1]
  
  
}
## 0) Pretreatment

Pretreatment = function(title_vec){
  titles = as.character(title_vec) %>%
    lapply(gsub, pattern = "[^[:alnum:][:space:]]",replacement = "") %>%
    lapply(replace_number) %>%
    lapply(stemDocument) %>%
    lapply(tolower)
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

## 3) Sentence Converter

###### 3a) word puller
Vector_puller = function(word, emb_matrix){
  if(word %in% rownames(emb_matrix)){
    return(emb_matrix[word,])
  }
  else{
    return(rep(0, ncol(emb_matrix)))
  }
}

Sentence_Vector = function(sentence, emb_matrix, stopwords){
  words = strsplit(sentence, " ", fixed = TRUE)[[1]]
  vec = lapply(words, Vector_puller, emb_matrix)
  df = matrix(unlist(vec), ncol = 50, byrow = T)
  zeros = which(rowSums(df) == 0)
  if(length(zeros) == 0)
  {
    df = apply(df, 2, mean)
    return(df)
  }
  else if(length(zeros) == nrow(df)){
    return(rep(0, 50))
  }
  else if(length(zeros) == nrow(df)-1){
    return(df[-zeros,])
  }
  else{
    df = apply(df[-zeros,], 2, mean)
    return(df)
  }
}



# Testing
library(e1071)
library(LilRhino)
library(forcats)
subs = c("physics", "bio", "med", "geo", "chem", "astro", "eng")
askscience_Data$tag = askscience_Data$tag %>%
  fct_collapse("Other" = c(as.character(unique(askscience_Data$tag)[which(!(unique(askscience_Data$tag) %in% subs))])))

askscience_Data$Title = unlist(Pretreatment(askscience_Data$Title))
#These suck I need better data
emb_mat = Embedding_Matrix(askscience_Data$Title, 5L, stopwords, 3L, 50, 100)

dat = Cross_val_maker(askscience_Data, .2)
test_vecs = dat$Test$Title %>%
  lapply(., Sentence_Vector, t(emb_mat), stopwords) %>%
  unlist %>%
  matrix(ncol = 50, byrow=T) %>%
  cbind(dat$Test$tag) %>%
  data.frame()

train_vecs = dat$Train$Title %>%
  Pretreatment() %>%
  unlist() %>%
  lapply(., Sentence_Vector, t(emb_mat), stopwords) %>%
  unlist() %>%
  matrix(ncol = 50, byrow=T)

