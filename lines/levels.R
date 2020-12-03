Gauss <- function(Sigma, mu, x) 
  {
  chisl <- exp((-1/2) %*% t(x - mu) %*% solve(Sigma) %*% (x - mu))
  znam <- sqrt(det(Sigma) * (2 * pi)^2)
  res <- chisl/znam
  return(res)
}

Sigma <- matrix(NA, 2, 2)
mu <- c(0, 0)

Sigma[1, 1] <- 2
Sigma[1, 2] <- 1
Sigma[2, 1] <- 1
Sigma[2, 2] <- 2

par(bg = 'White', fg = 'black')
plot(x, y, type = "n",asp = 1)

for (i in seq(-5, 5, 0.1)) 
  {
  for (j in seq(-5, 5, 0.1)) 
    {
    color <- adjustcolor("yellow", Gauss(Sigma, mu, c(i, j)) * 2)
    points(i, j, pch = 21,col = color, bg = color)
  }
}

x <- seq(-5, 5, 0.1)
y <- seq(-5, 5, 0.1)

z <- outer(x, y, function(x, y) {
  sapply(1:length(x), function(i) Gauss(Sigma, mu, c(x[i], y[i])))
})

contour(x,y,z,add = T ,asp = 1,lwd = 1)
