euclideanDistance <- function(u, v)
{
  sqrt(sum((u - v)^2))
}
sortObjectsByDist <- function(x1, z, metricFunction =
                                euclideanDistance)
  {
  l <- dim(x1)[1]
  n <- dim(x1)[2]-1
  distances <- matrix(NA, l, 2)
  for (i in 1:l) 
    {
    distances[i, ] <- c(i, euclideanDistance(x1[i, 1:n], z)) 
  }
  orderedX1 <- x1[order(distances[, 2]), ]
  return(orderedX1)
}
kNN <- function(k, orderedX1)
  {
  n <- dim(orderedX1)[2] - 1
  classes <- orderedX1[1:k, n + 1]
  counts <- table(classes)
  class <- names(which.max(counts))
  return(class)
}
LOO <- function(x1)
  {
  m <- dim(x1)[1]
  l <- m - 1
  L <- matrix(0, l, 1)
  for (i in 1:m) 
    {
    z <- x1[i, 1:2]
    x2 <- x1[-i, ]
    orderedX2 <- sortObjectsByDist(x2, z)
    for (k in 1:l) 
      {
      class <- kNN(k, orderedX2)
      if (class != x1[i, 3]) 
        {
        L[k] <- L[k] + 1
      }
    }
  }
  min_k <- which.min(L[1:l])
  min_m <- L[min_k]
  I <- matrix(1:l, l, 1)
  for (i in 1:l)
    {
    L[i] <- L[i]/l
  }
  plot(I[1:l], L[1:l], type = "s", xlab = "k", ylab = "LOO")
  points(min_k, min_m/l, pch = 21, bg = "black")
  text(min_k, min_m/l,labels=min_k, cex= 0.7, pos=3)
  return(min_k)
}
  k <- LOO(iris[, 3:5])
  colors <- c("setosa" = "red", "versicolor" = "green3",
              "virginica" = "blue")
  plot(iris[, 3:4], pch = 21, bg = colors[iris$Species], col
       = colors[iris$Species], asp = 1)
  