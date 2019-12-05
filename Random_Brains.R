RB = function(data, y, x_test, variables = ceiling(ncol(data)/10),  brains = floor(sqrt(ncol(data))), hiddens = c(3, 4)){
  # Data: Dataframe or matrix of values
  data = as.data.frame(cbind(data, y))
  colnames(data) = c(paste('V', 1:(ncol(data)-1), sep = ''), 'label')
  preds = matrix(NA, ncol = brains, nrow = nrow(x_test))
  final_preds = c()
  cols = matrix(ncol = variables,
                nrow = brains)
  for(i in 1:brains){
    coldex = sample(1:(ncol(data)-1), variables)
    cols[i,] = sort(coldex)
    res = neuralnet(label~., data = data[,c(coldex, ncol(data))], hidden = hiddens, linear.output = F)
    preds[,i] = apply(predict(res, x_test[,coldex]), 1, which.max)
  }
  for(i in 1:nrow(preds)){
    final_preds = c(final_preds, names(head(sort(table(preds[i,])),1)))
  }
  return(list('predictions' = final_preds, 'num_brains' = brains, 'predictors_per_brain' = variables,
              'hidden_layers' = hiddens, preds_per_tree = cols))
}




lalaland = Cross_val_maker(iris, .2)

latrain = lalaland$Train
latest = lalaland$Test

Final_Test = RB(latrain[,-5], latrain$Species, latest[,-5], latest$Species, variables = 2, brains = 3)
table(Final_Test, latest$Species)
