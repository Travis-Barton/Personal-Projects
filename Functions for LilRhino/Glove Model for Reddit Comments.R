# Glove model creation for raddit comments

library(DBI)
require(RSQLite)
library(dblyr)

db <- src_sqlite('database.sqlite', create = F)
db_subset <- db %>%
  tbl('May2015') %>%
  data.frame()

