# ML1
Способ представления исходных данных - матрица объектов признака. По столбца номера признаковб на пересечении значения признаков для текущего объекта.

Метод обучения строит по выборке алгоритм отображения.

Классический метод обучения по прецеденту - минимизация эмпирического риска.

Алгоритм ближайшего соседа - 1NN относит классифицируемый объект к тому классу, к которму принадлежит его ближайший сосед.
```R
oneNN <- function(xl, z)
{
  orderedXl <- sortObjectsByDist(xl, z)
  n <- dim(orderedXl)[2] - 1
  classes <- orderedXl[1, n + 1]
  counts <- table(classes)
  class <- names(which.max(counts))
  return (class)
}
```
![1nn](https://github.com/kristinaovc/ML1/blob/master/1NN.PNG)

Алгоритм k ближайших соседей - kNN относит объект к тому классу, элементов которого больше среди k ближайших соседей
```R
kNN <- function(xl, z, k)
{
  orderedXl <- sortObjectsByDist(xl, z, k)
  n <- dim(orderedXl)[2] - 1
  classes <- orderedXl[1:k, n + 1]	
  counts <- table(classes)	
  class <- names(which.max(counts))	
  return (class)	
}
```
![knn](https://github.com/kristinaovc/ML1/blob/master/kNN.PNG)

Оптимальное k подбирается по критерию скользящего контроля LOO.

Алгоритм k взвешенных ближайших соседей - kwNN в каждом классе выбирается k ближайших объектов, и объект относится к тому классу, для которого среднее расстояние до k ближайших соседей минимально.
