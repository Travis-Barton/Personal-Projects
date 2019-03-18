# THE FOLLOWING WAS WRITTEN BY DR CRISTINA TORTORA #
####################################################

# Ratio written by Calinski and Harbasz to find the optimal Kmeans cluster number

ratio.strength <- function(X,G,output){ # A function to compute Calinski and  Harabasz  ratio 
  n <- nrow(X) # computes the number of observations
  sst <- cov.wt(X)$cov*(n-1) # computes the SST matrix
  # print(sst)
  ssw <- 0 # sets the initial SSW to be 0
  for(g in 1:G){ # for each group g
    mu.g <- output$center[g,] # isolate the mean vector of interest
    partition <- which(g == output$cluster) # gives the relevant observations determined from allocate
    ng <- length(partition) # counts the number of observations in each group
    ssw <- ssw + cov.wt(X[partition,],center=mu.g)$cov*(ng-1) # computes the ssw for group g and sums over all g
  }
  ssb <- sst - ssw  # computes SSB matrix
  val <- (sum(diag(ssb))*(n-G))/(sum(diag(ssw))*(G-1)) # Computes the ratio given on Slide 11
  return(val)
}