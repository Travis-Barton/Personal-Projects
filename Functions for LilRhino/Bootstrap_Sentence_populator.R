library(readr)
library(magrittr)
library(dplyr)
library(LilRhino)
library(parallel)
testing_set = c(paste('this is test',  as.character(seq(1, 10, 1))))

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



# from your testng, you've seen that it does well in enhancing
# a binary decision. But it doesn't seem to do well with The whole
# data set. This needs more investigation.

# It appears no new information is being added. Perhaps the synonym 
# idea should re-appear. 




##################### Testint ######################

Bootstrap_Vocab(testing_set, 3, c('is'))


data = read_csv('Askscience_Data_body.csv')[,-1]

stops = Pretreatment(data$Title, F, T, T) %>%
  unlist() %>%
  Stopword_Maker(100)

blah = filter(data, tag == 'maths') %>%
  extract(,2) %>%
  as.matrix() %>%
  Pretreatment(F, T) %>%
  unlist() %>%
  Bootstrap_Vocab(3, stopwds = stops)
blah2 = data %>% extract(,2) %>%
  as.matrix() %>%
  Pretreatment(F, T) %>%
  unlist()

blah2 = Bootstrap_Data_Frame(blah2, data$tag, stops)
