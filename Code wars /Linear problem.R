#Linear challange

s <- function(x)
{
  return(2*x+1)
}
f <- function(x)
{
  return(3*x+1)
}

sn <- function(x)
{
  if(x != 1)
  {return()}
}

dblLinear <- function (k) {
  k = k+1
  s <- function(x)
  {
    return(2*x+1)
  }
  f <- function(x)
  {
    return(3*x+1)
  }
  n = 1
  while(length(n) < k)
  {
    sn = {}
    fn = {}
    for(i in 1:length(n))
    {
      sn = c(sn, s(n[i]))
      fn = c(fn, f(n[i]))
    }
    temp = sort(c(sn, fn))
    index <- which(temp > tail(n, 1))
    temp <- temp[index]
    n = c(n, temp[1])
  }
  return(n)
}

(test <- dblLinear(20))


dblLinear2 <- function (k) {
  k = k+1
  s <- function(x)
  {
    return(2*x+1)
  }
  f <- function(x)
  {
    return(3*x+1)
  }
  n = c(1, 3, 4)
  j = 1
  i = 1
  while(length(n) < k)
  {
    
    m <- tail(n, 1)
    while(f(n[j]) <= m)
    {
      j = j+1
      
    }
    while(s(n[i]) <= m)
    {
      i = i+1
      
    }
   # print(i)
 
    #print(j)
   #print(c(s(i), f(i)))
    canidate <- min(c(s(n[i]), f(n[j])))
    
    #temp = sort(c(sn, fn))
    ##index <- which(temp > tail(n, 1))
    #temp <- temp[index]
    n = c(n, canidate)
  }
  return(n[k])
}

(test <- dblLinear2(20))


