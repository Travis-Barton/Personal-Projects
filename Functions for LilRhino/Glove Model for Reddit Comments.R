# Glove model creation for raddit comments

library(DBI)
require(RSQLite)
library(dplyr)
library(parallel)
library(progress)
setwd("~/Downloads")
con <- dbConnect(RSQLite::SQLite(), "database.sqlite")
df = dbGetQuery(con, 'SELECT body FROM May2015 LIMIT 2000000')
dbDisconnect(con)
numcores = detectCores()

titles = as.character(df$body)


### New apporach
cleaner = function(vec){
  return(as.character(vec %>%
    replace_emoji() %>%
    replace_url() %>%
    replace_contraction() %>%
    Num_Al_sep() %>%
    gsub(pattern = "[^[:alnum:][:space:]]", replacement = "")))
}

i = 1
temp = {}
out = {}
while(i <= 10000){
  temp = titles[i:(i+5000)] %>%
    mclapply(cleaner) %>%
    unlist
  i = i+5000
  out = c(out, temp)
  print(i)
}


