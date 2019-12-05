dict_maker = function(corpi){
  dict = list()
  corpi = strsplit(corpi, ' ')
  lapply(corpi, function(doc){
      for(i in 1:(length(doc)-1)){
      if(doc[i] %in% names(dict)){
        dict[[doc[i]]] <<- c(dict[[doc[i]]], doc[i+1])
      }
      else{
        dict[[doc[i]]] <<- doc[i+1]
      }}
    })
   return(dict)
 }
Bootstrap_Vocab2 = function(vocab, N, stopwds, dict, min_length = 7, max_length = 15){
  res = {}
  vocab = strsplit(vocab, ' ')
  while(length(res) < N){
    seed = sample(vocab, 1)
    if(seed[length(seed)] %in% names(dict)){
      seed = c(seed, sample(dict[[seed]], 1))
    }
    else{
      
    }
  }
}






dict = dict_maker(test_sent)
test_sent = c('this is a test for potatos', 'i like for potatos')






Bootstrap_Vocab = function(vocab, N, stopwds, min_length = 7, max_length = 15)
{
  
  res = {}
  sent = ''
  cutoff = sample(min_length:max_length, 1)
  while(length(res) < N){
    sent = paste(sent, vocab[sample(1:length(vocab), 1)] %>%
                   strsplit(' ') %>%
                   unlist() %>%
                   data.frame('words' = ., stringsAsFactors = F) %>%
                   filter(!(words %in% stopwds)) %>%
                   sample_n(1), ' ')
    
    len = strsplit(sent, ' ') %>%
      unlist() %>%
      as.data.frame() %>%
      filter(. != '') %>%
      nrow()
    
    
    if(len > cutoff){
      res = c(res, sent)
      sent = ''
      cutoff = sample(min_length:max_length, 1)
    }
    
  }
  return(res)
} # For one class

Bootstrap_Data_Frame = function(text, tags, stopwords, min_length = 7, max_length = 15)
{
  max_tags =floor(1.1*max(table(tags)))
  newdata = data.frame('text' = text, 'tags' = tags, stringsAsFactors = F)
  for (tag in unique(tags)) {
    tag_index = which(tags == tag)
    num_to_boostrap = max_tags - length(tag_index)
    new_sents = text[tag_index] %>%
      Bootstrap_Vocab(num_to_boostrap, stopwords, min_length, max_length)
    
    new_row = cbind(new_sents, rep(tag, length(new_sents)))
    new_row = data.frame('text' = new_row[,1], 'tags' = new_row[,2], stringsAsFactors = F)
    newdata = rbind(newdata, new_row)
  }
  newdata$text = as.character(newdata$text)
  return(newdata)
} # For the whole database
