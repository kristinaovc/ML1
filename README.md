# ML1

## Метрические алгоритмы классификации

### Метод ближайших соседей

Для произвольного объекта ![1](https://github.com/kristinaovc/ML1/blob/master/image/1.PNG) расположим элементы обучающей выборки ![2](https://github.com/kristinaovc/ML1/blob/master/image/2.PNG) в порядке возрастания расстояния до ![3](https://github.com/kristinaovc/ML1/blob/master/image/3.PNG):

![4](https://github.com/kristinaovc/ML1/blob/master/image/4.PNG)

где ![5](https://github.com/kristinaovc/ML1/blob/master/image/5.PNG)— i-ый сосед объекта u, а ![6](https://github.com/kristinaovc/ML1/blob/master/image/6.PNG)— ответ на i-ом соседе.

**Метрический алгоритм классификации с обучающей выборкой ![7](https://github.com/kristinaovc/ML1/blob/master/image/7.PNG) относит объект ![3](https://github.com/kristinaovc/ML1/blob/master/image/3.PNG) к тому классу ![8](https://github.com/kristinaovc/ML1/blob/master/image/8.PNG) , для которого суммарный вес ближайших обучающих объектов ![9](https://github.com/kristinaovc/ML1/blob/master/image/9.PNG) максимален:**

 ![10](https://github.com/kristinaovc/ML1/blob/master/image/10.PNG)
 
 где весовая функция ![11](https://github.com/kristinaovc/ML1/blob/master/image/11.PNG) оценивает степень важности i-го соседа для классификации объекта ![3](https://github.com/kristinaovc/ML1/blob/master/image/3.PNG). Функция ![12](https://github.com/kristinaovc/ML1/blob/master/image/12.PNG) — называется оценкой близости объекта ![3](https://github.com/kristinaovc/ML1/blob/master/image/3.PNG) к классу y.

Обучающая выборка ![7](https://github.com/kristinaovc/ML1/blob/master/image/7.PNG)  играет роль параметра алгоритма. Алгоритм ![13](https://github.com/kristinaovc/ML1/blob/master/image/13.PNG) строит локальную аппроксимацию выборки ![7](https://github.com/kristinaovc/ML1/blob/master/image/7.PNG), причем вычисления откладываются до момента, пока не станет известен объект ![3](https://github.com/kristinaovc/ML1/blob/master/image/3.PNG). По этой причине метрические алгоритмы относятся к методам ленивого обучения (lazy learning), в отличие от усердного обучения (eager learning), когда на этапе обучения строится функция, аппроксимирующая выборку.

Метрические алгоритмы классификации относятся также к методам рассуждения по прецедентам (case-based reasoning, CBR). Классический метод обучения по прецеденту - минимизация эмпирического риска.

Способ представления исходных данных - матрица объектов признака. По столбца номера признаковб на пересечении значения признаков для текущего объекта.

Метод обучения строит по выборке алгоритм отображения.

#### **Алгоритм ближайшего соседа - 1NN** относит классифицируемый объект ![14](https://github.com/kristinaovc/ML1/blob/master/image/14.PNG) к тому классу, к которму принадлежит его ближайший сосед:

![15](https://github.com/kristinaovc/ML1/blob/master/image/15.PNG)

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
![klass1nn](https://github.com/kristinaovc/ML1/blob/master/klass1nn.PNG)

#### **Алгоритм k ближайших соседей - kNN** относит объект u к тому классу, элементов которого больше среди k ближайших соседей ![5](https://github.com/kristinaovc/ML1/blob/master/image/5.PNG) , ![16](https://github.com/kristinaovc/ML1/blob/master/image/16.PNG)

![17](https://github.com/kristinaovc/ML1/blob/master/image/17.PNG)


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
![knn](https://github.com/kristinaovc/ML1/blob/master/images/kknn.PNG)
![klassknn](https://github.com/kristinaovc/ML1/blob/master/klassknn.PNG)

*Вопрос:* Как выбирать k? При k = 1 получаем метод ближайшего соседа и, соответственно, неустойчивость к шуму, при k = l, наоборот, алгоритм чрезмерно устойчив и вырождается в константу. Таким образом, крайние значения k нежелательны. На практике оптимальное k подбирается по критерию скользящего контроля LOO(leave one out) и CV (Cross Validation).

#### **Алгоритм LOO:** убираем из выборки один i-тый элемент. Все остальные становятся обучающими, а I-тый контрольным. и смотрим ответы на обучающих и н контрольном. Если они отличаются, то Loo=1.

![18](https://github.com/kristinaovc/ML1/blob/master/image/18.PNG) - выборка

![19](https://github.com/kristinaovc/ML1/blob/master/image/19.PNG) - ответы

![20](https://github.com/kristinaovc/ML1/blob/master/image/20.PNG) - выборка(алгоритм)

Возьмем алгоритм и проверим на ![21](https://github.com/kristinaovc/ML1/blob/master/image/21.PNG)

![22](https://github.com/kristinaovc/ML1/blob/master/image/22.PNG)

то есть

![23](https://github.com/kristinaovc/ML1/blob/master/image/23.PNG) - суммируем и делим на l

LOO считает для метода по выборке ![24](https://github.com/kristinaovc/ML1/blob/master/image/24.PNG)

Обучающая способность алгоритма проверяется на данных, которые не использовались при обучении.

```R
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
```

![LOO](https://github.com/kristinaovc/ML1/blob/master/LOO.PNG)

### **Алгоритм k взвешенных ближайших соседей - kwNN** в каждом классе выбирается k ближайших объектов, и объект относится к тому классу, для которого среднее расстояние до k ближайших соседей минимально.

![25](https://github.com/kristinaovc/ML1/blob/master/image/25.PNG)

![26](https://github.com/kristinaovc/ML1/blob/master/image/26.PNG)— строго убывающая последовательность вещественных весов, задающая вклад i-го соседа при классификации объекта ![3](https://github.com/kristinaovc/ML1/blob/master/image/3.PNG).

**Примеры весов** ![27](https://github.com/kristinaovc/ML1/blob/master/image/27.PNG) — геометрическая прогрессия со знаменателем ![28](https://github.com/kristinaovc/ML1/blob/master/image/28.PNG) , который можно подбирать по критерию LOO.

![29](https://github.com/kristinaovc/ML1/blob/master/kwnn.PNG)

**Недостатки:**

- Приходится хранить обучающую выборку целиком. Это приводит к неэффективному расходу памяти и чрезмерному усложнению решающего правила. При наличии погрешностей (как в исходных данных, так и в модели сходства ρ это может приводить к понижению точности классификации вблизи границы классов. Имеет смысл отбирать минимальное подмножество эталонных объектов, действительно необходимых для классификации.
- Поиск ближайшего соседа предполагает сравнение классифицируемого объекта со всеми объектами выборки за O(l) операций. Для задач с большими выборками или высокой частотой запросов это может оказаться накладно. Проблема решается с помощью эффективных алгоритмов поиска ближайших соседей, требующих в среднем O(ln l) операций.
- В простейших случаях метрические алгоритмы имеют крайне бедный набор параметров, что исключает возможность настройки алгоритма по данным.

### Метод парзеновского окна

