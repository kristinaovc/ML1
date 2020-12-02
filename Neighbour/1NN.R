euclideanDistance <- function(u, v)
{
  sqrt(sum((u - v)^2))
}
sortObjectsByDist <- function(xl, z)
{
  l <- dim(xl)[1]
  n <- dim(xl)[2] - 1
  distances <- matrix(NA, l, 2)
  for (i in 1:l)
  {
    distances[i, ] <- c(i,euclideanDistance(xl[i, 1:n], z))
  }
  orderedXl <- xl[order(distances[, 2]), ]
  return (orderedXl);
}
oneNN <- function(xl, z)
{
  orderedXl <- sortObjectsByDist(xl, z)
  n <- dim(orderedXl)[2] - 1
  classes <- orderedXl[1, n + 1]
  counts <- table(classes)
  class <- names(which.max(counts))
  return (class)
}
colors <- c("setosa" = "red", "versicolor" = "green3",
            "virginica" = "blue")
plot(iris[, 3:4], pch = 21, bg = colors[iris$Species], col
     = colors[iris$Species], asp = 1)
z <- c(2.3, 0.5)
xl <- iris[, 3:5]
class <- oneNN(xl, z)
points(z[1], z[2], pch = 21, bg = colors[class], asp = 1)
for (i in seq(0.8, 7.2, 0.1)) {
  for (j in seq(-1, 3.5, 0.1)) {
    points(i, j, pch = 1, col = colors[oneNN(iris[, 3:5], c(i, j))])
  }
}