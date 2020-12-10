naive <- function(x, Py, mu, sigma, m, n) 
  {
  amo <- matrix(c('setosa','versicolor', 'virginica', 0, 0, 0), nrow = 3, ncol = 2)
  scores <- rep(0, m)
  for (i in 1:m) 
    {
    scores[i] <- Py[i]
    for (j in 1:n)
      {
      N <- 1/sqrt(2*pi)/sigma[i,j]*exp(-1/2*(x[j]-mu[i,j])^2/sigma[i,j]^2)
      scores[i] <- scores[i] * N
    }
    amo[i,2] <-  scores[i]
  }
  class <- amo[,1][which.max(amo[,2])]
}

xl <- iris[, 3:5]
n<-2 
m<-3 
classes <- levels(xl[,3])
Py<-table(xl[,3])/dim(xl)[1]

mu <- matrix(0, nrow=m, ncol=n)
sigma <- matrix(0, nrow=m, ncol=n)
for(i in 1:m)
  {
  for(j in 1:n)
    {
    temp<-xl[xl[,3]==classes[i],][,j] 
    mu[i,j]<-mean(temp)
    sigma[i,j]<-sqrt(var(temp))
  }
}

colors <- c("setosa" = "red", "versicolor" = "green3", "virginica" = "blue")
plot(iris[, 3:4], pch = 21, bg = colors[iris$Species], col = colors[iris$Species], asp = 1, main = "Наивный байесовский классификатор")

for (i in seq(-1.2, 3.8, 0.1))
  {
  for( j in seq(0.8, 7.2, 0.1))
    {
     z <- c(j, i)
    class <- naive(z, Py, mu, sigma, m, n)
    points(z[1], z[2], pch = 21, col = colors[class], asp = 1)
  }
}