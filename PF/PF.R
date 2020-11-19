install.packages("plotrix")
require("plotrix")

euclideanDistance <- function(u, v)
{
  sqrt(sum((u - v)^2))
}

kernel.R <- function(r){
  0.5 * (abs(r) <= 1)# Прямоугольное ядро
} 
kernel.T <- function(r){
  (1 - abs(r)) * (abs(r) <= 1) # Треугольное ядро
}
kernel.Q <- function(r){
  (15 / 16) * (1 - r ^ 2) ^ 2 * (abs(r) <= 1) # Квартическое ядро
}
kernel.E <- function(r){
  (3 / 4) * (1 - r ^ 2) * (abs(r) <= 1) # Ядро Епанечникова
}
kernel.G <- function(r){
  (2 * pi) ^ (-0.5) * exp(-0.5 * (r ^ 2)) # Гауссовское ядро
} 

getNewSetByPotentials <- function(xl, potentials, res) 
  {
  if(!is.null(ncol(res))) return (res[as.numeric(rownames(xl[which(potentials != 0),])), ]) 
  return (res[as.numeric(rownames(xl[which(potentials != 0),]))]) 
}

distToObjects <- function(xl, z, metricFunction = euclideanDistance) 
  {
  l <- nrow(xl)
  n <- ncol(xl)
  distances <- rep(0, l)
  for (i in 1:l) 
    distances[i] <- metricFunction(xl[i, 1:(n-1)], z)
  return (distances)
}

findH <- function(xl) 
  {
  l <- nrow(xl)
  h <- rep(0, l)
  for(i in 1:l) 
    { 
    h[i] <- 0.4 
  }
  return (h)
}

getPotentials <- function(xl, h, eps, main_kernel)
{
  l <- nrow(xl) 
  n <- ncol(xl) 
  potentials <- rep(0, l)
  distances_to_points <- matrix(0, l, l)
  err <- eps + 1 
  for (i in 1:l)
    distances_to_points[i,] <- distToObjects(xl, c(xl[i, 1], xl[i, 2])) 
  while(err > eps)
    {
    while (TRUE) 
      {
      cur <- sample(1:l, 1)
      class <- PF(distances_to_points[cur, ], potentials, h, xl, main_kernel)
      if (class != xl[cur, n]) 
        {
        potentials[cur] = potentials[cur] + 1
        break
      } 
    } 
    err <- 0
    for (i in 1:l) 
      {
      class <- PF(distances_to_points[i, ], potentials, h, xl, main_kernel)
      err <- err + (class != xl[i, n])
    }
  }
  return (potentials)
}

PF <- function(distances, potentials, h, xl, main_kernel) 
  {
  l <- nrow(xl)
  n <- ncol(xl)
  classes <- xl[, n]
  weights <- table(classes) 
  weights[1:length(weights)] <- 0
  for (i in 1:l) 
    { 
    class <- xl[i, n] 
    r <- distances[i] / h[i]
    weights[class] <- weights[class] + potentials[i] * main_kernel(r) 
  }
  if (max(weights) != 0) return (names(which.max(weights)))
  return (0)
}

ClassMap <- function(xl, h, potentials, main_kernel) 
  {
  classMatrix <- matrix(NA, length(seq(0.8, 7.2, 0.1))*length(seq(-1, 3.2, 0.1)), ncol(xl))
  cnt <- 1
  for (i in seq(0.8, 7.2, 0.1))
  {
    for (j in seq(-1, 3.2, 0.1)) 
      {
      z <- c(i, j)
      distances <- distToObjects(xl, z)
      class <- PF(distances, potentials, h, xl, main_kernel)
      if (class != 0) 
        {
        classMatrix[cnt, ] <- c(z[1], z[2], class)
        cnt <- cnt + 1
      }
    }
  }
  return (classMatrix)
}

drawPlots <- function(xl, classMatrix, potentials, h) 
  {
  l <- nrow(xl)
  n <- ncol(xl)
  colors <- c("setosa" = "red", "versicolor" = "green3", "virginica" = "blue")
  transp_red <- col2rgb("red")
  transp_red <- rgb(transp_red[1], transp_red[2], transp_red[3], alpha = 60, max = 255)
  transp_green <- col2rgb("green3")
  transp_green <- rgb(transp_green[1], transp_green[2], transp_green[3], alpha = 60, max = 255)
  transp_blue <- col2rgb("blue")
  transp_blue <- rgb(transp_blue[1], transp_blue[2], transp_blue[3], alpha = 60, max = 255)
  transp_colors <- c("setosa" = transp_red, "versicolor" = transp_green, "virginica" = transp_blue)
  par(mfrow=c(1,2))

  plot(xl[, 1:(n-1)], pch = 21, bg = colors[xl[,n]], col = colors[xl[,n]], main = "Карта потенциалов", asp = 1)
  for (i in 1:l)
    if (potentials[i] != 0)
      draw.circle(xl[i, 1], xl[i, 2], radius = potentials[i], border = transp_colors[xl[i, n]], col = transp_colors[xl[i, n]])
  plot(xl[, 1:(n-1)], pch = 21, bg = colors[xl[,n]], col = colors[xl[,n]], main = "Метод потенциальных функций",  asp = 1)
  points(classMatrix[, 1:(n-1)], pch = 1, col = colors[classMatrix[, n]])
}

main <- function(main_kernel) 
  {
  xl <- iris[, 3:5]
  h <- findH(xl)
  potentials <- getPotentials(xl, h, 5, main_kernel)
  new_xl <- getNewSetByPotentials(xl, potentials, xl) 
  new_h <- getNewSetByPotentials(xl, potentials, h) 
  new_potentials <- getNewSetByPotentials(xl, potentials, potentials) 
  classMatrix <- ClassMap(new_xl, new_h, new_potentials, main_kernel)
  drawPlots(xl, classMatrix, potentials, h)
}

main(kernel.R)
