
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
  test = unlist(lapply(as.vector(titles), strsplit, split = ' ', fixed = TRUE))
  stopwords = test %>%
    table() %>%
    sort(decreasing = TRUE) %>%
    head(cutoff) %>%
    names()
  return(stopwords)
}


## 2) Auto-creating word matrix


Embedding_Matrix = function(sent, vocab_min, stopwords, skip_gram, vector_size){
  it   = itoken(unlist(sent), tolower, word_tokenizer)
  vocab = create_vocabulary(it, stopwords = stopwords)
  vocab <- prune_vocabulary(vocab, term_count_min = vocab_min)
  # Use our filtered vocabulary
  vectorizer <- vocab_vectorizer(vocab)
  # use window of 5 for context words
  tcm <- create_tcm(it, vectorizer, skip_grams_window = skip_gram)
  glove = GlobalVectors$new(word_vectors_size = vector_size, vocabulary = vocab, x_max = 5)
  glove$fit_transform(tcm, n_iter = 20)
  return(glove$components)
}
colnames(temp)

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
  #  for(i in 1:4){
  #    Vector_puller(word[i], emb_matrix)
  #
  #  }
  vec = lapply(words, Vector_puller, emb_matrix)
  df <- data.frame(matrix(unlist(vec), nrow=ncol(emb_matrix), byrow=F))
  return(rowSums(df)/length(unique(colSums(df))))
}



# Testing
library(e1071)
test_titles = unlist(Pretreatment(dat$Test$Title))

test_vecs = dat$Test$Title %>%
  Pretreatment() %>%
  lapply(., Sentence_Vector, word_vecs, stopwords) %>%
  unlist %>%
  matrix(nrow = length(dat$Test$Title), byrow=T) %>%
  cbind(dat$Test$tag) %>%
  data.frame()

train_vecs = dat$Train$Title %>%
  Pretreatment() %>%
  unlist() %>%
  lapply(., Sentence_Vector, word_vecs, stopwords) %>%
  unlist() %>%
  matrix(nrow = length(dat$Train$Title), byrow=T) %>%
  cbind(dat$Train$tag) %>%
  data.frame() 

fitSVM = svm(X51~., data = train_vecs, type = 'C-classification', kernel = )
sum(predict(fitSVM, test_vecs[,-51]) == test_vecs$X51)/898

sum(predict(fitSVM, train_vecs[,-51]) == train_vecs$X51)/5089



FeedData = Feed_Reduction(as.matrix(train_vecs[,-51]), train_vecs$X51, as.matrix(test_vecs[,-51]))



FeedTrain = FeedData$train
FeedTest = FeedData$test

annoying = cbind(rbind(FeedTrain, FeedTest), c(train_vecs$X51, test_vecs$X51))
annoying = as.data.frame(annoying)
Feedmod = svm(V9~., data = annoying, subset= c(1:5089), type = 'C-classification')
sum(predict(Feedmod, annoying[c(1:5089),]) == train_vecs$X51)/length(train_vecs$X51)
sum(predict(Feedmod, annoying[-c(1:5089),]) == test_vecs$X51)/length(test_vecs$X51)
