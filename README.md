# ML1

## Метрические алгоритмы классификации

### Метод ближайших соседей

Для произвольного объекта ![1](https://github.com/kristinaovc/ML1/blob/master/images/1.PNG) расположим элементы обучающей выборки ![2](https://github.com/kristinaovc/ML1/blob/master/images/2.PNG) в порядке возрастания расстояния до u:

![3](https://github.com/kristinaovc/ML1/blob/master/images/3.PNG)

где ![4](https://github.com/kristinaovc/ML1/blob/master/images/4.PNG)— i-ый сосед объекта u, а ![5](https://github.com/kristinaovc/ML1/blob/master/images/5.PNG)— ответ на i-ом соседе.

**Метрический алгоритм классификации с обучающей выборкой ![6](https://github.com/kristinaovc/ML1/blob/master/images/6.PNG) относит объект u к тому классу ![7](https://github.com/kristinaovc/ML1/blob/master/images/7.PNG) , для которого суммарный вес ближайших обучающих объектов ![8](https://github.com/kristinaovc/ML1/blob/master/images/8.PNG) максимален:**

 ![9](https://github.com/kristinaovc/ML1/blob/master/images/9.PNG)
 
 где весовая функция ![10](https://github.com/kristinaovc/ML1/blob/master/images/10.PNG) оценивает степень важности i-го соседа для классификации объекта u. Функция ![11](https://github.com/kristinaovc/ML1/blob/master/images/11.PNG) — называется оценкой близости объекта u к классу y.


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
