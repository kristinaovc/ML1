euclideanDistance <- function(u, v)
{
  sqrt(sum((u - v)^2))
}

distToObjects <- function(points, u)
  {
  apply(points, 1, euclideanDistance, u)
  }


#Ядра
kernel.R <- function(r){
  0.5 * (abs(r) <= 1) #прямоугольное
} 
kernel.T <- function(r){
  (1 - abs(r)) * (abs(r) <= 1) #треугольное
  }
kernel.Q <- function(r){
  (15 / 16) * (1 - r ^ 2) ^ 2 * (abs(r) <= 1) #квартическое
}
kernel.E <- function(r){
  (3 / 4) * (1 - r ^ 2) * (abs(r) <= 1) #епанечниково
}
kernel.G <- function(r){
  dnorm(r) #гауссовское 
} 

PW <- function(distToObjects, u, h) 
  {
  weights <- kernel.G(distToObjects / h)
  classes <- unique(names(distToObjects))
  wSort <- sapply(classes, function(class, arr) { sum(arr[names(arr) == class]) } , weights)
  if (max(wSort) == 0)
    {
    return("") 
  }
  return(names(which.max(wSort)))
}

LOO <- function(x1, class, Val) 
  {
  n <- dim(x1)[1]
  loo <- rep(0, length(Val))
  for (i in 1:n) 
    {
    u <- x1[i,]
    z <- x1[-i,]
    distToObjects <- distToObjects(z, u)
    names(distToObjects) <- class[-i]
    for (j in 1:length(Val)) 
      {
      h <- Val[j]
      classified <- PW(distToObjects, u, h)
      loo[j] <- loo[j] + (classified != class[i])
    }
  }
  loo <- loo / n
}

drawLOO <- function(x1, class, Val) 
  {
  loo <- LOO(x1, class, Val)
  x <- Val
  y <- loo
  plot(x, y, type = "l", main = "LOO гауссовское ядро", xlab = "h", ylab = "LOO")
  h <- Val[which.min(loo)]
  h.loo <- round(loo[which.min(loo)], 4)
  points(h, h.loo, pch = 19, col = "red")
  label <- paste("h = ", h, "\n", "LOO = ", h.loo, sep = "")
  text(h, h.loo, labels = label, pos = 3, col = "red")
  return(h)
}

drawPW <- function(x1, class, colors, h) 
  {
  uniqueClass <- unique(class)
  names(colors) <- uniqueClass
  plot(x1, bg = colors[class], pch = 21, asp = 1,main = "Карта классификации ") 
  for (i in seq(0.8, 7.2, 0.1)) 
    {
    for (j in seq(-1, 3.2, 0.1)) 
      {
      i <- round(i, 1) 
      j <- round(j, 1) 
      z <- c(i, j)
      distToObjects <- distToObjects(points, z)
      names(distToObjects) <- class
      classified <- PW(distToObjects, z, h)
      points(z[1], z[2], col = colors[classified], pch = 21)
    }
  }
}

par(mfrow <- c(1, 2))
h <- drawLOO(iris[, 3:4], iris[, 5], hValues = seq(0.1, 2, 0.005))
drawPW(iris[, 3:4], iris[, 5], colors = c("setosa" = "red", "versicolor" = "green3", "virginica" = "blue"), h = h)
