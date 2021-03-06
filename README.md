# ML1
  # Оглавление
1. [Метрические алгоритмы классификации](#Metrical)
    1. [Алгоритм 1NN](#1NN)
    2. [Алгоритм KNN](#KNN)
    3. [Алгоритм KWNN](#KWNN)
    4. [Алгоритм парзеновкого окна](#PW)
    5. [Алгоритм потенциальных функций](#PF)
    6. [Алгоритм STOLP](#STOLP)
2. [Байесовские	алгоритмы	классификации](#Bayes)
    1. [Линии уровня](#Lines)
    2. [Наивный байесовский классификатор](#Class)
    3. [Подстановочный алгоритм(Plug-In)](#PlugIn)
    4. [Линейный дискриминант Фишера](#LDF)
3. [Линейные алгоритмы	классификации](#Linear)
    1. [ADALINE. Правило	Хэбба](#Adaline)
    2. [Логистическая	регрессия](#Regression)

| Алгоритм| k/h | Величина ошибки |
| :---: | :---: | :---: |
| KNN| k=6 | 0.034 |
| KWNN| k=9 | 0.033 |
| PW Прямоугольное ядро| h=0.32| 0.04 |
| PW Треугольное ядро | h=0.32| 0.04 |
| PW Квартическое ядро| h=0.32| 0.04 |
| PW Епанечникова| h=0.32| 0.04 |
| PW Гауссовское ядро| h=0.1| 0.04 |
| PF все| h=1| 0.046 |

## Метрические алгоритмы классификации <a name="Metrical"></a>

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

#### **Алгоритм ближайшего соседа - 1NN** относит классифицируемый объект ![14](https://github.com/kristinaovc/ML1/blob/master/image/14.PNG) к тому классу, к которму принадлежит его ближайший сосед: <a name="1NN"></a>

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
![1nn](https://github.com/kristinaovc/ML1/blob/master/Neighbour/1NN.PNG)
![klass1nn](https://github.com/kristinaovc/ML1/blob/master/Neighbour/klass1nn.PNG)

#### **Алгоритм k ближайших соседей - kNN** относит объект u к тому классу, элементов которого больше среди k ближайших соседей <a name="KNN"></a> ![5](https://github.com/kristinaovc/ML1/blob/master/image/5.PNG) , ![16](https://github.com/kristinaovc/ML1/blob/master/image/16.PNG)

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
![klassknn](https://github.com/kristinaovc/ML1/blob/master/Neighbour/klassknn.PNG)

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

![LOO](https://github.com/kristinaovc/ML1/blob/master/Neighbour/LOO.PNG)

### **Алгоритм k взвешенных ближайших соседей - kwNN** в каждом классе выбирается k ближайших объектов, и объект относится к тому классу, для которого среднее расстояние до k ближайших соседей минимально. <a name="KWNN"></a>

![25](https://github.com/kristinaovc/ML1/blob/master/image/25.PNG)

![26](https://github.com/kristinaovc/ML1/blob/master/image/26.PNG)— строго убывающая последовательность вещественных весов, задающая вклад i-го соседа при классификации объекта ![3](https://github.com/kristinaovc/ML1/blob/master/image/3.PNG).

**Примеры весов** ![27](https://github.com/kristinaovc/ML1/blob/master/image/27.PNG) — геометрическая прогрессия со знаменателем ![28](https://github.com/kristinaovc/ML1/blob/master/image/28.PNG) , который можно подбирать по критерию LOO.

```R
kwNN <- function(xl, z, k, q)
{
  orderedXl <- sortObjectsByDist(xl, z)
  n <- dim(orderedXl)[2] - 1
  for(i in 1:k){
    orderedXl[i, 4] <- q^i
  }
  types <- c("setosa", "versicolor", "virginica")
  mat <- matrix(data=0, nrow=1, ncol=3)
  names(mat) <- types
  classes <- orderedXl[1:k, (n+1):(n+2)]
  mat[1,1] <- sum(classes[classes$Species=="setosa",2])
  mat[1,2] <- sum(classes[classes$Species=="versicolor",2])
  mat[1,3] <- sum(classes[classes$Species=="virginica",2])
  count <- which.max(mat)
  class <- types[count]
  return (class)
}
```

![29](https://github.com/kristinaovc/ML1/blob/master/Neighbour/kwnn.PNG)
![30](https://github.com/kristinaovc/ML1/blob/master/Neighbour/klasskwnn.PNG)
```R
LOO = function(xl) 
  {
  n <- dim(xl)[1]-1
  loo <- rep(0, n)
  class <- iris[, 5]
  for(i in 1:(n+1))
    {
    X <- xl[-i, 1:3]
    z <- xl[i, 1:2]
    orderedXl <- sortObjectByDist(X, z)
    for(k in 1:n)
      {
      test <- kwnn(X,z,k,orderedXl)
      if(colors[test] != colors[class[i]]){
        loo[k] <- loo[k]+1;
      }    
    } 
  }
  loo <- loo/(n+1)
  x <- 1:(length(loo))
  y <- loo
  plot(x, y,main ="LOO for KWNN(k)", xlab="k", ylab="LOO", type = "l")
  min <- which.min(loo)
  lOOmin <- round(loo[min],3)
  points(min, loo[min], pch = 21, col = "red",bg = "red")
  text(lOOmin, labels =paste("K = ", min), cex= 0.7, pos=3, col = "red")
}
```
![30](https://github.com/kristinaovc/ML1/blob/master/Neighbour/lookwnn.PNG)

## Сравнение kNN И kwNN

| Алгоритм| k= | Величина ошибки |
| :---: | :---: | :---: |
| kNN | 6 | 0.034 |
| kwNN | 9 | 0.033 |

Метод k ближайших соседей. Для повышения надёжности классификации объект относится к тому классу, которому принадлежит большинство из его соседей — k ближайших к нему объектов обучающей выборки x_i. В задачах с двумя классами число соседей берут нечётным, чтобы не возникало ситуаций неоднозначности, когда одинаковое число соседей принадлежат разным классам.

Метод взвешенных ближайших соседей. В задачах с числом классов 3 и более нечётность уже не помогает, и ситуации неоднозначности всё равно могут возникать. Тогда i-му соседу приписывается вес w_i, как правило, убывающий с ростом ранга соседа i. Объект относится к тому классу, который набирает больший суммарный вес среди k ближайших соседей.

**Недостатки:**

- Приходится хранить обучающую выборку целиком. Это приводит к неэффективному расходу памяти и чрезмерному усложнению решающего правила. При наличии погрешностей (как в исходных данных, так и в модели сходства ρ это может приводить к понижению точности классификации вблизи границы классов. Имеет смысл отбирать минимальное подмножество эталонных объектов, действительно необходимых для классификации.
- Поиск ближайшего соседа предполагает сравнение классифицируемого объекта со всеми объектами выборки за O(l) операций. Для задач с большими выборками или высокой частотой запросов это может оказаться накладно. Проблема решается с помощью эффективных алгоритмов поиска ближайших соседей, требующих в среднем O(ln l) операций.
- В простейших случаях метрические алгоритмы имеют крайне бедный набор параметров, что исключает возможность настройки алгоритма по данным.

### Метод парзеновского окна <a name="PW"></a>

Рассмотрим весовую функцию ![11](https://github.com/kristinaovc/ML1/blob/master/image/11.PNG) как функцию не от ранга соседа, а как функцию от расстояния ![29](https://github.com/kristinaovc/ML1/blob/master/image/29.PNG) :

![30](https://github.com/kristinaovc/ML1/blob/master/image/30.PNG) ,

где ![31](https://github.com/kristinaovc/ML1/blob/master/image/31.PNG) — невозрастающая на ![32](https://github.com/kristinaovc/ML1/blob/master/image/32.PNG) (гипотеза компактности) функция ядра. В этом случае метрический классификатор примет следующий вид:

![33](https://github.com/kristinaovc/ML1/blob/master/image/33.PNG)

Алгоритм ![34](https://github.com/kristinaovc/ML1/blob/master/image/34.PNG)  называется алгоритмом парзеновского окна.

Параметр h называется шириной окна и играет примерно ту же роль, что и число соседей k. “Окно” — это сферическая окрестность объекта ![3](https://github.com/kristinaovc/ML1/blob/master/image/3.PNG) радиуса h, при попадании в которую обучающий объект ![21](https://github.com/kristinaovc/ML1/blob/master/image/21.PNG) “голосует” за отнесение объекта ![3](https://github.com/kristinaovc/ML1/blob/master/image/3.PNG) к классу ![35](https://github.com/kristinaovc/ML1/blob/master/image/35.PNG).

Параметр h можно задавать априори или определять по скользящему контролю. Зависимость LOO(h), как правило, имеет характерный минимум, поскольку слишком узкие окна приводят к неустойчивой классификации; а слишком широкие — к вырождению алгоритма в константу.

Если объекты существенно неравномерно распределены по пространству X, то необходимо использовать метод парзеновского окна с переменной шириной окна:

![36](https://github.com/kristinaovc/ML1/blob/master/image/36.PNG).

Чтобы найти ширину окна и наиболее подходящий нам тип ядра, мы воспользуемся принципом максимального правдоподобия и исключением объектов по одному leave-one-out:

![37](https://github.com/kristinaovc/ML1/blob/master/image/37.PNG).

То есть, мы будем восстанавливать значение класса для одного объекта из нашей выборки и максимизировать логарифм количества правильных ответов при исключении по очереди всех объектов выборки. Максимизация этого значения происходит по двум параметрам - ширине окна h и типу ядерной функции. Ширину окна мы можем подобрать из некоторого диапазона \delta{}H, полученного из эмпирических предположений. Ядро выбирается из нижеприведенного набора ядер:

Прямоугольное 

![42](https://github.com/kristinaovc/ML1/blob/master/image/42.PNG).

```R
kernel.R <- function(r){
  0.5 * (abs(r) <= 1) 
}
```

![43](https://github.com/kristinaovc/ML1/blob/master/PW/PWPryamoug.PNG).

Треугольное

![40](https://github.com/kristinaovc/ML1/blob/master/image/40.PNG).

```R
kernel.T <- function(r){
  (1 - abs(r)) * (abs(r) <= 1) 
  }
```

![44](https://github.com/kristinaovc/ML1/blob/master/PW/PWTreug.PNG).

Квартическое

![39](https://github.com/kristinaovc/ML1/blob/master/image/39.PNG).

```R
kernel.Q <- function(r){
  (15 / 16) * (1 - r ^ 2) ^ 2 * (abs(r) <= 1) 
}
```

![45](https://github.com/kristinaovc/ML1/blob/master/PW/PWKvart.PNG).

Епанечникова

![38](https://github.com/kristinaovc/ML1/blob/master/image/38.PNG).

```R
kernel.E <- function(r){
  (3 / 4) * (1 - r ^ 2) * (abs(r) <= 1) 
}
```

![46](https://github.com/kristinaovc/ML1/blob/master/PW/PWepanech.PNG).

Гауссовское

![41](https://github.com/kristinaovc/ML1/blob/master/image/41.PNG).

```R
kernel.G <- function(r){
  dnorm(r) 
} 
```

![46](https://github.com/kristinaovc/ML1/blob/master/PW/PWgauss.PNG).

```R
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
```
### Метод потенциальных функций <a name="PF"></a>

Если в методе парзеновского окна центр окна поместить в классифицируемый объект, то получим метод потенциальных функций:

![43](https://github.com/kristinaovc/ML1/blob/master/image/43.PNG)

Теперь ширина окна зависит не от классифицируемого объекта ![44](https://github.com/kristinaovc/ML1/blob/master/image/44.PNG), а от обучающего ![21](https://github.com/kristinaovc/ML1/blob/master/image/21.PNG).

Данный алгоритм имеет достаточно богатый набор из ![45](https://github.com/kristinaovc/ML1/blob/master/image/45.PNG) параметров ![46](https://github.com/kristinaovc/ML1/blob/master/image/46.PNG)

**Алгоритм**

**Вход**: ![7](https://github.com/kristinaovc/ML1/blob/master/image/7.PNG)— обучающая выборка

**Выход**: Коэффициенты ![47](https://github.com/kristinaovc/ML1/blob/master/image/47.PNG)

1: Инициализация: ![48](https://github.com/kristinaovc/ML1/blob/master/image/48.PNG)

2: **повторять**

3: выбрать объект ![49](https://github.com/kristinaovc/ML1/blob/master/image/49.PNG)

4: **если** ![50](https://github.com/kristinaovc/ML1/blob/master/image/50.PNG)  **то**

5:  ![51](https://github.com/kristinaovc/ML1/blob/master/image/51.PNG) 

6: **пока** число ошибок на выборке не окажется достаточно мало

```R
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
```

Ядро Епанечникова

![52](https://github.com/kristinaovc/ML1/blob/master/PF/PFEpanech.PNG)

Гауссовское ядро

![53](https://github.com/kristinaovc/ML1/blob/master/PF/PFGauss.PNG)

Квартическое ядро

![54](https://github.com/kristinaovc/ML1/blob/master/PF/PFKvart.PNG)

Прямоугольное ядро

![55](https://github.com/kristinaovc/ML1/blob/master/PF/PFPryamoug.PNG)

Треугольное ядро

![56](https://github.com/kristinaovc/ML1/blob/master/PF/PFTreug.PNG)

**Преимущества метода потенциальных функций:**

- Метод прост для понимания и алгоритмической реализации;

- Порождает потоковый алгоритм;

- Хранит лишь часть выборки, следовательно, экономит память.

**Недостатки метода:**

- Порождаемый алгоритм медленно сходится;

- Параметры ![46](https://github.com/kristinaovc/ML1/blob/master/image/46.PNG) настраиваются слишком грубо;

- Значения параметров ![47](https://github.com/kristinaovc/ML1/blob/master/image/47.PNG) зависят от порядка выбора объектов из выборки ![7](https://github.com/kristinaovc/ML1/blob/master/image/7.PNG).

### Алгоритм STOLP <a name="STOLP"></a>

Отступом (margin) объекта ![21](https://github.com/kristinaovc/ML1/blob/master/image/21.PNG) относительно алгоритма классификации, имеющего вид ![52](https://github.com/kristinaovc/ML1/blob/master/image/52.PNG), называется величина

![53](https://github.com/kristinaovc/ML1/blob/master/image/53.PNG)

Отступ показывает степень типичности объекта. Отступ отрицателен тогда и только тогда, когда алгоритм допускает ошибку на данном объекте. В зависимости от значений отступа обучающие объекты условно делятся на пять типов, в порядке убывания отступа: эталонные, неинформативные, пограничные, ошибочные, шумовые:

- Эталонные объекты имеют большой положительный отступ, плотно окружены объектами своего класса и являются наиболее типичными его представителями.

- Неинформативные объекты также имеют положительный отступ. Изъятие этих объектов из выборки (при условии, что эталонные объекты остаются), не влияет на качество классификации. Фактически, они не добавляют к эталонам никакой новой информации. Наличие неинформативных объектов характерно для выборок избыточно большого объема.

- Пограничные объекты имеют отступ, близкий к нулю. Классификация таких объектов неустойчива в том смысле, что малые изменения метрики или состава обучающей выборки могут изменять их классификацию.

- Ошибочные объекты имеют отрицательные отступы и классифицируются неверно. Возможной причиной может быть неадекватность алгоритмической модели, в частности, неудачная конструкция метрики ρ.

- Шумовые объекты или выбросы — это небольшое число объектов с большими отрицательными отступами. Они плотно окружены объектами чужих классов и классифицируются неверно. Они могут возникать из-за грубых ошибок или пропусков в исходных данных, а также по причине отсутствия важной информации, которая позволила бы отнести эти объекты к правильному классу.

**Алгоритмы STOLP**

**Вход:** ![7](https://github.com/kristinaovc/ML1/blob/master/image/7.PNG)  — обучающая выборка,

![54](https://github.com/kristinaovc/ML1/blob/master/image/54.PNG) — порог фильтрации выбросов,
 
![55](https://github.com/kristinaovc/ML1/blob/master/image/55.PNG) — допустимая доля ошибок.

**Выход:** Множество опорных объектов ![56](https://github.com/kristinaovc/ML1/blob/master/image/56.PNG)

1: **для всех** ![49](https://github.com/kristinaovc/ML1/blob/master/image/49.PNG) проверить, является ли ![21](https://github.com/kristinaovc/ML1/blob/master/image/21.PNG) выбросом:

2: **если**  ![57](https://github.com/kristinaovc/ML1/blob/master/image/57.PNG) **то**

3: ![58](https://github.com/kristinaovc/ML1/blob/master/image/58.PNG)

4: : Взять по одному эталону из каждого класса: // Инициализация

![59](https://github.com/kristinaovc/ML1/blob/master/image/59.PNG)

5: **пока** ![60](https://github.com/kristinaovc/ML1/blob/master/image/60.PNG)

6: Выделить множество объектов, на которых алгоритм ![61](https://github.com/kristinaovc/ML1/blob/master/image/61.PNG) ошибается:

![62](https://github.com/kristinaovc/ML1/blob/master/image/62.PNG)

7: **если** ![63](https://github.com/kristinaovc/ML1/blob/master/image/63.PNG) **то**

8: **выход**

9: Присоединить к ![64](https://github.com/kristinaovc/ML1/blob/master/image/64.PNG) объект с наименьшим отступом:

![65](https://github.com/kristinaovc/ML1/blob/master/image/65.PNG)

Результатом работы алгоритма STOLP является разбиение обучающих объектов на три категории: шумовые, эталонные и неинформативные. Если гипотеза компактности верна и выборка достаточно велика, то основная масса обучающих объектов окажется неинформативной и будет отброшена. Фактически, этот алгоритм выполняет сжатие исходных данных

```R
margin -> function(points,classes,point,class)
{
  Myclass -> points[which(classes==class), ]
  OtherClass -> points[which(classes!=class), ]
  MyMargin -> Parzen(Myclass,point[1:2],1,FALSE)
  OtherMargin -> Parzen(OtherClass,point[1:2],1,FALSE)
  return(MyMargin-OtherMargin)
}
```

![66](https://github.com/kristinaovc/ML1/blob/master/Stolp/StolpMargin.PNG)


![67](https://github.com/kristinaovc/ML1/blob/master/Stolp/StolpPoints.PNG)

![68](https://github.com/kristinaovc/ML1/blob/master/Stolp/StolpMap.PNG)

## Байесовские алгоритмы классификации <a name="Bayes"></a>

Байесовский подход является классическим в теории распознавания образов и лежит в основе многих методов. Он опирается на теорему о том, что если плотности распределения классов известны, то алгоритм классификации, имеющий миинимальную вероятность ошибок, можно выписать в явном виде.

### Линии уровня <a name="Lines"></a>

Случайная величина ![66](https://github.com/kristinaovc/ML1/blob/master/image/66.PNG) имеет многомерное нормальное распределение с параметрами ![67](https://github.com/kristinaovc/ML1/blob/master/image/67.PNG) и ![68](https://github.com/kristinaovc/ML1/blob/master/image/68.PNG)  если ее плотность задается выражением

![69](https://github.com/kristinaovc/ML1/blob/master/image/69.PNG)

Параметр ![70](https://github.com/kristinaovc/ML1/blob/master/image/70.PNG) является мат.ожиданием, а ![71](https://github.com/kristinaovc/ML1/blob/master/image/71.PNG) — матрицей ковариации нормального распределения. Матрица ![71](https://github.com/kristinaovc/ML1/blob/master/image/71.PNG) является симметричной и положительно определенной

Линии уровня плотности нормального распределения соответствуют линиям уровня квадратичной формы ![72](https://github.com/kristinaovc/ML1/blob/master/image/72.PNG) и представляют собой эллипсы. Можно выделить три основных вида этих линий уровня в зависимости от значения матрицы ковариации. Если матрица ковариации пропорциональна единичной, то все компоненты нормального распределения ![21](https://github.com/kristinaovc/ML1/blob/master/image/21.PNG) являются независимыми друг от друга и имеют одинаковую дисперсию. Линии уровня при этом образуют окружности. Диагональная матрица ковариации соответствует независимым компонентам ![21](https://github.com/kristinaovc/ML1/blob/master/image/21.PNG), но с различными дисперсиями. Линии уровня в этом случае являются эллипсами, параллельными координатным осям. Наконец, произвольная положительная определенная матрица ковариации соответствует эллипсам общего вида.

**1)Признаки некоррелированы, с одинаковыми десперсиями. Линии уровня соответствуют окружностям.**

![73](https://github.com/kristinaovc/ML1/blob/master/image/73.PNG)

![76](https://github.com/kristinaovc/ML1/blob/master/lines/level1.PNG)

**2)Признаки некоррелированы, с различными десперсиями.Линии уровня соответствуют эллипсу параллельному осям координат.**

![74](https://github.com/kristinaovc/ML1/blob/master/image/74.PNG)

![77](https://github.com/kristinaovc/ML1/blob/master/lines/level2.PNG)

**3)Признаки коррелированы, с различными десперсиями. Линии уровня соответствуют эллипсам общего вида.**

![75](https://github.com/kristinaovc/ML1/blob/master/image/75.PNG)

![78](https://github.com/kristinaovc/ML1/blob/master/lines/level3.PNG)

### Наивный байесовский классификатор <a name="CLass"></a>

Будем полагать, что все объекты описываются ![76](https://github.com/kristinaovc/ML1/blob/master/image/76.PNG) числовыми признаками. Обозначим через ![77](https://github.com/kristinaovc/ML1/blob/master/image/77.PNG) произвольный элемент пространства объектов ![78](https://github.com/kristinaovc/ML1/blob/master/image/78.PNG), где ![79](https://github.com/kristinaovc/ML1/blob/master/image/79.PNG).

Предположим, что все признаки ![80](https://github.com/kristinaovc/ML1/blob/master/image/80.PNG)  являются независимыми случайными величинами. Следовательно, функции правдоподбия классов представимы в виде

![81](https://github.com/kristinaovc/ML1/blob/master/image/81.PNG)

где ![82](https://github.com/kristinaovc/ML1/blob/master/image/82.PNG) — плотность распределений значений j-го признака для класса y.

Оценивать ![76](https://github.com/kristinaovc/ML1/blob/master/image/76.PNG) дномерных плотностей гораздо проще, чем одну n-мерную
плотность. Однако данное предположение крайне редко работает на практике, поэтому алгоритмы, использующие ![80](https://github.com/kristinaovc/ML1/blob/master/image/80.PNG), называют наивными байесовскими.

Подставим эмпирические оценки одномерных плотностей ![82](https://github.com/kristinaovc/ML1/blob/master/image/82.PNG) в оптимальный байесовский классификатор. Получим алгоритм наивный байесовский классификатор 

![83](https://github.com/kristinaovc/ML1/blob/master/image/83.PNG)

![84](https://github.com/kristinaovc/ML1/blob/master/image/84.PNG)

![83](https://github.com/kristinaovc/ML1/blob/master/Naive/naive.PNG)

```R
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
```

**Преимущества**

• Простота реализации.

• Низкие вычислительные затраты при обучении и классификации.

• В тех редких случаях, когда признаки (почти) независимы, наивный байесовский классификатор (почти) оптимален.

**Недостатки**

• Низкое качество классификации. Он используется либо как эталон при экспериментальном сравнении алгоритмов, либо как элементарный «строительный блок» при алгоритмических композициях.

### Подстановочный алгоритм Plug-In <a name="PlugIn"></a>

Оценим параметры функций правдоподобия ![85](https://github.com/kristinaovc/ML1/blob/master/image/85.PNG) и ![86](https://github.com/kristinaovc/ML1/blob/master/image/86.PNG) по частям обучающей выборки ![87](https://github.com/kristinaovc/ML1/blob/master/image/87.PNG) для каждого класса ![8](https://github.com/kristinaovc/ML1/blob/master/image/8.PNG). Затем эти выборочные оценки подставим в оптимальный байесовский классификатор. Получим байесовский нормальный классификатор, который называется также подстановочным (plug-in).

![88](https://github.com/kristinaovc/ML1/blob/master/image/88.PNG)

![89](https://github.com/kristinaovc/ML1/blob/master/image/89.PNG)

1. Разделяющая кривая - эллипс.

![89](https://github.com/kristinaovc/ML1/blob/master/PlugIn/plug1.PNG)

2. Разделяющая кривая - парабола.

![90](https://github.com/kristinaovc/ML1/blob/master/PlugIn/plug3.PNG)

3. Разделяющая кривая - гипербола.

![91](https://github.com/kristinaovc/ML1/blob/master/PlugIn/plug2.PNG)

```R
plot(xl[,1], xl[,2], pch = 21, bg = colors[xl[,3]], asp = 1, xlab = "Ось X", ylab = "Ось Y", main = "Подстановочный алгоритм(Plug-in)") 

objectsOfFirstClass <- xl[xl[,3] == 1, 1:2] 
objectsOfSecondClass <- xl[xl[,3] == 2, 1:2] 

mu1 <- estimateMu(objectsOfFirstClass) 
mu2 <- estimateMu(objectsOfSecondClass) 
sigma1 <- estimateCovarianceMatrix(objectsOfFirstClass, mu1) 
sigma2 <- estimateCovarianceMatrix(objectsOfSecondClass, mu2) 
coeffs <- getPlugInDiskriminantCoeffs(mu1, sigma1, mu2, sigma2) 
 
x <- y <- seq(-10, 20, len=100)  
z <- outer(x, y, function(x, y) coeffs["x^2"]*x^2 + coeffs["xy"]*x*y + coeffs["y^2"]*y^2 + coeffs["x"]*x + coeffs["y"]*y + coeffs["1"])  
contour(x, y, z, levels=0, drawlabels=FALSE, lwd = 3, col = "red", add = TRUE)

```
Недостатки подстановочного алгоритма вытекают из нескольких чрезмерно сильных базовых предположений, которые на практике часто не выполняются.

- Функции правдоподобия классов могут существенно отличаться от гауссовских. В частности, когда имеются признаки, принимающие дискретные значения, или когда классы распадаются на изолированные сгустки.

- Если длина выборки меньше размерности пространства, ![95](https://github.com/kristinaovc/ML1/blob/master/image/95.PNG), или среди признаков есть линейно зависимые, то матрица ![86](https://github.com/kristinaovc/ML1/blob/master/image/86.PNG) становится вырожденной. В этом случае обратная матрица не существует и метод вообще неприменим.

- На практике встречаются задачи, в которых признаки «почти линейно зависимы». Тогда матрица ![86](https://github.com/kristinaovc/ML1/blob/master/image/86.PNG) является плохо обусловленной, то есть близкой к некоторой вырожденной матрице. Это так называемая проблема мультиколлинеарности, которая влечет неустойчивость обратной матрицы ![96](https://github.com/kristinaovc/ML1/blob/master/image/96.PNG). Она может непредсказуемо и сильно изменяться при незначительных вариациях исходных данных, например, связанных с погрешностями измерений. Неустойчивость снижает качество классификации.

- Выборочные оценки чувствительны к нарушениям нормальности распределений, в частности, к редким большим выбросам.

### Линейный дискриминант Фишера <a name="LDF"></a>

Пусть ковариационные матрицы классов одинаковы и равны ![71](https://github.com/kristinaovc/ML1/blob/master/image/71.PNG). Оценим ![91](https://github.com/kristinaovc/ML1/blob/master/image/91.PNG) по всем ![92](https://github.com/kristinaovc/ML1/blob/master/image/92.PNG) объектам обучающей выборки. С учетом поправки на смещённость,

![93](https://github.com/kristinaovc/ML1/blob/master/image/93.PNG)

В этом случае разделяющая поверхность линейна (кусочно-линейна). Подстановочный алгоритм имеет вид:

![94](https://github.com/kristinaovc/ML1/blob/master/image/94.PNG)

Этот алгоритм называется линейным дискриминантом Фишера (ЛДФ). Он неплохо работает, когда формы классов действительно близки к нормальным и не слишком сильно различаются. В этом случае линейное решающее правило близко к оптимальному байесовскому, но существенно более устойчиво, чем квадратичное, и часто обладает лучшей обобщающей способностью.

![94](https://github.com/kristinaovc/ML1/blob/master/LDF/LDF.PNG)

```R
LDF <- function(Py, lambda, n, m, mu, sigma, point) 
  {
  point <- as.numeric(point)
  p <- rep(0, m)
  for (i in 1:m) 
    {
    p[i] <- Py[i] * lambda[i]
    p[i] <- p[i] *  exp(-(1/2) * t(point - mu[i, ]) %*% solve(sigma) %*% (point - mu[i, ])) / sqrt((2 * pi)^n * det(sigma))
  }
  return(classes[which.max(p)])
}
```
Эвристика линейного дискриминанта Фишера является в некотором роде упрощением квадратичного дискриминанта. Она используется с целью получить более устойчивый алгоритм классификации. Наиболее целесообразно пользоваться линейным дискриминантом Фишера, когда данных для обучения недостаточно. Вследствие основной гипотезы, на которой базируется алгоритм, наиболее удачно им решаются простые задачи классификации, в которых по формам классы "похожи" друг на друга.

## Линейные алгоритмы классификации <a name="Linear"></a>

Пусть ![78](https://github.com/kristinaovc/ML1/blob/master/image/78.PNG) и ![97](https://github.com/kristinaovc/ML1/blob/master/image/97.PNG).

Если дискриминантная функция определяется как скалярное произведение вектора ![98](https://github.com/kristinaovc/ML1/blob/master/image/98.PNG) и вектора пераметров ![99](https://github.com/kristinaovc/ML1/blob/master/image/99.PNG) , то получается линейный
классификатор:

![100](https://github.com/kristinaovc/ML1/blob/master/image/100.PNG)


Уравнение ![101](https://github.com/kristinaovc/ML1/blob/master/image/101.PNG)  задает гиперплоскость, разделяющую классы в пространстве ![102](https://github.com/kristinaovc/ML1/blob/master/image/102.PNG). Если вектор ![98](https://github.com/kristinaovc/ML1/blob/master/image/98.PNG)  находится по одну сторону гиперплоскости с ее направляющим вектором ![103](https://github.com/kristinaovc/ML1/blob/master/image/103.PNG) , то объект ![98](https://github.com/kristinaovc/ML1/blob/master/image/98.PNG)  относится к классу +1, иначе — к классу −1.

#### Метод стохастического градиента

Пусть задана обучающая выборка ![104](https://github.com/kristinaovc/ML1/blob/master/image/104.PNG). Требуется найти вектор параметров ![99](https://github.com/kristinaovc/ML1/blob/master/image/99.PNG), при котором достигается минимум аппроксимированного эмпирического риска:

![105](https://github.com/kristinaovc/ML1/blob/master/image/105.PNG)


Применим для минимизаци ![106](https://github.com/kristinaovc/ML1/blob/master/image/106.PNG)  метод градиентного спуска. В этом методе выбирается некоторое начальное приближение для ![103](https://github.com/kristinaovc/ML1/blob/master/image/103.PNG),  затем запускается итерационный процесс, на каждом шаге которого вектор ![103](https://github.com/kristinaovc/ML1/blob/master/image/103.PNG)  изменяется в направлении наиболее быстрого убывания функционала ![107](https://github.com/kristinaovc/ML1/blob/master/image/107.PNG).  Это направление противоположно направлению вектора градиента ![108](https://github.com/kristinaovc/ML1/blob/master/image/108.PNG):

![109](https://github.com/kristinaovc/ML1/blob/master/image/109.PNG),

где ![110](https://github.com/kristinaovc/ML1/blob/master/image/110.PNG) — темп обучения.

Предположим, что функция ![111](https://github.com/kristinaovc/ML1/blob/master/image/111.PNG) дифференцируема. Выпишем градиент:

![112](https://github.com/kristinaovc/ML1/blob/master/image/112.PNG)

Каждый прецедент ![113](https://github.com/kristinaovc/ML1/blob/master/image/113.PNG)  вносит аддитивный вклад в изменение вектора ![103](https://github.com/kristinaovc/ML1/blob/master/image/103.PNG), но вектор ![103](https://github.com/kristinaovc/ML1/blob/master/image/103.PNG)  изменяется только после перебора всех ![92](https://github.com/kristinaovc/ML1/blob/master/image/92.PNG) объектов. Сходимость итерационного процесса можно улучшить, если выбирать прецеденты ![113](https://github.com/kristinaovc/ML1/blob/master/image/113.PNG) по одному, для каждого делать градиентный шаг и сразу обновлять вектор весов:

![114](https://github.com/kristinaovc/ML1/blob/master/image/114.PNG)


В методе стохастического градиента (stochastic gradient, SG) прецеденты перебираются в случайном порядке. Если же объекты предъявлять в некотором фиксированном порядке, процесс может зациклиться или разойтись.

**Вход** : ![7](https://github.com/kristinaovc/ML1/blob/master/image/7.PNG) — обуч. выборка; η — темп обучения; ![115](https://github.com/kristinaovc/ML1/blob/master/image/115.PNG) — параметр сглаживания.

**Выход** : Веса ![116](https://github.com/kristinaovc/ML1/blob/master/image/116.PNG).

1: Инициализировать веса ![116](https://github.com/kristinaovc/ML1/blob/master/image/116.PNG).

2: Вычислить начальное значение функционала ![117](https://github.com/kristinaovc/ML1/blob/master/image/117.PNG).

3: **повторять**

4: выбрать объект ![49](https://github.com/kristinaovc/ML1/blob/master/image/49.PNG).

5: вычислить ошибку алгоритма:

![118](https://github.com/kristinaovc/ML1/blob/master/image/118.PNG)

6: сделать шаг градиентного спуска:

![114](https://github.com/kristinaovc/ML1/blob/master/image/114.PNG)

7: оценить новое значение функционала:

![119](https://github.com/kristinaovc/ML1/blob/master/image/119.PNG)

8: **пока** значение Q не стабилизируется и/или веса w не перестанут изменяться.

```R
margins <- array(dim = l)
    for (i in 1:l)
    {
      xi <- xl[i, 1:n]
      yi <- xl[i, n + 1]
      margins[i] <- crossprod(w, xi) * yi 
    }
    errorIndexes <- which(margins <= 0) 
    if (length(errorIndexes) > 0)
    {
      i <- sample(1:l, 1)
      iterCount <- iterCount + 1
      xi <- xl[i, 1:n]
      yi <- xl[i, n + 1]
      wx <- crossprod(w, xi)
      margin <- wx * yi
      ex <- habb(margin)
      w <- w + eta * yi * xi 
      Qprev <- Q
      Q <- (1 - lambda) * Q + lambda * ex 
    } else
    {
      break
    }
```

**Преимущества метода SG**

- Метод легко реализуется и легко обобщается на нелинейные классификаторы и на нейронные сети — суперпозиции линейных классификаторов.

- Метод подходит для динамического обучения, когда обучающие объекты поступают потоком, и вектор весов обновляется при появлении каждого объекта.

- Метод позволяет настраивать веса на избыточно больших выборках, за счет того, что случайной подвыборки может оказаться достаточно для обучения.

### ADALINE.	Правило	Хэбба <a name="Adaline"></a>

Возьмем ![120](https://github.com/kristinaovc/ML1/blob/master/image/120.PNG). Тогда ![121](https://github.com/kristinaovc/ML1/blob/master/image/121.PNG), где производная берется по ![103](https://github.com/kristinaovc/ML1/blob/master/image/103.PNG), вычисляя которую

![122](https://github.com/kristinaovc/ML1/blob/master/image/122.PNG)

получим правило обновления весов на каждой итерации метода стохастического градиента:

![123](https://github.com/kristinaovc/ML1/blob/master/image/123.PNG)

Это правило предложено Видроу и Хоффом и называется дельта-правилом, а сам линейный нейрон — **адаптивным линейным элементом (ADALINE)**.

```R

adaline <- function(x)
{
  return ((x-1)^2)
} 

```

```R
l <- dim(xl)[1]
  n <- dim(xl)[2] - 1
  w <- c(1/2, 1/2, 1/2)
  iterCount <- 0
  Q <- 0
  for (i in 1:l) {
    wx <- sum(w * xl[i, 1:n])
    margin <- wx * xl[i, n + 1]
    Q <- Q + adaline(margin)
  }
  repeat 
    {
    margins <- array(dim = l)
    for (i in 1:l)
    {
      xi <- xl[i, 1:n]
      yi <- xl[i, n + 1]
      margins[i] <- crossprod(w, xi) * yi 
    }
    errorIndexes <- which(margins <= 0) 
    if (length(errorIndexes) > 0)
    {
      i <- sample(1:l, 1)
      iterCount <- iterCount + 1
      xi <- xl[i, 1:n]
      yi <- xl[i, n + 1]
      wx <- crossprod(w, xi)
      margin <- wx * yi
      ex <- adaline(margin)
      w <- w - eta * (wx - yi) * xi
      Qprev <- Q
      Q <- (1 - lambda) * Q + lambda * ex 
    } else
    {
      break
    }
  }
  return(w) 
```

![123](https://github.com/kristinaovc/ML1/blob/master/Adaline/adaline1.PNG)

#### Правило Хэбба

Будем считать признаки бинарными: ![124](https://github.com/kristinaovc/ML1/blob/master/image/124.PNG). Тогда при классификации ![125](https://github.com/kristinaovc/ML1/blob/master/image/125.PNG) объекта ![21](https://github.com/kristinaovc/ML1/blob/master/image/21.PNG) возможны следующие три случая:

1. Если ответ ![125](https://github.com/kristinaovc/ML1/blob/master/image/125.PNG) совпадает с истинным ![35](https://github.com/kristinaovc/ML1/blob/master/image/35.PNG), то вектор весов изменять не надо.
 
2. Если ![126](https://github.com/kristinaovc/ML1/blob/master/image/126.PNG) и ![127](https://github.com/kristinaovc/ML1/blob/master/image/127.PNG), то вектор весов ![103](https://github.com/kristinaovc/ML1/blob/master/image/103.PNG) увеличивается (можно только те ![128](https://github.com/kristinaovc/ML1/blob/master/image/128.PNG) , для которых ![129](https://github.com/kristinaovc/ML1/blob/master/image/129.PNG)) ![130](https://github.com/kristinaovc/ML1/blob/master/image/130.PNG), где ![110](https://github.com/kristinaovc/ML1/blob/master/image/110.PNG) — темп обучения.

3. Если ![131](https://github.com/kristinaovc/ML1/blob/master/image/131.PNG) и ![132](https://github.com/kristinaovc/ML1/blob/master/image/132.PNG), то вектор весов w уменьшается: ![133](https://github.com/kristinaovc/ML1/blob/master/image/133.PNG).

Эти три случая объединяются в так называемое правило Хэбба:

**если** ![134](https://github.com/kristinaovc/ML1/blob/master/image/134.PNG) то ![135](https://github.com/kristinaovc/ML1/blob/master/image/135.PNG).

```R

habb <- function(x)
{
  return (max(-x, 0))
}

```

```R
 l <- dim(xl)[1]
  n <- dim(xl)[2] - 1
  w <- c(1/2, 1/2, 1/2)
  iterCount <- 0
  Q <- 0
  for (i in 1:l) 
    {
    wx <- sum(w * xl[i, 1:n])
    margin <- wx * xl[i, n + 1]
    Q <- Q + habb(margin)
  }
  repeat 
    {
    margins <- array(dim = l)
    for (i in 1:l)
    {
      xi <- xl[i, 1:n]
      yi <- xl[i, n + 1]
      margins[i] <- crossprod(w, xi) * yi 
    }
    errorIndexes <- which(margins <= 0) 
    if (length(errorIndexes) > 0)
    {
      i <- sample(1:l, 1)
      iterCount <- iterCount + 1
      xi <- xl[i, 1:n]
      yi <- xl[i, n + 1]
      wx <- crossprod(w, xi)
      margin <- wx * yi
      ex <- habb(margin)
      w <- w + eta * yi * xi 
      Qprev <- Q
      Q <- (1 - lambda) * Q + lambda * ex 
    } else
    {
      break
    }
  }
  return(w) 

```

![124](https://github.com/kristinaovc/ML1/blob/master/Adaline/habb1.PNG)

### Логистическая	регрессия <a name="Regression"></a>

Метод логистической регрессии основан на довольно сильных вероятностных предположениях, которые имеют несколько интересных последствий:

1. линейный классификатор оказывается оптимальным байесовским;

2. однозначно определяется функция потерь;

3. можно вычислять не только принадлежность объектов классам, но также получать и численные оценки вероятности их принадлежности.

Логистическая функция потерь ![136](https://github.com/kristinaovc/ML1/blob/master/image/136.PNG).

Логистическое правило обновления весов для градиентного шага в методе стохастического градиента: ![137](https://github.com/kristinaovc/ML1/blob/master/image/137.PNG).

![138](https://github.com/kristinaovc/ML1/blob/master/image/138.PNG) - — сигмоидная функция.

```R
loss <- function(x)
{
  return (log2(1 + exp(-x)))
}
sigmoid <- function(z)
{
  return (1 / (1 + exp(-z)))
}
```

```R
 l <- dim(xl)[1]
  n <- dim(xl)[2] - 1
  w <- c(1/2, 1/2, 1/2)
  iterCount <- 0
  Q <- 0
  for (i in 1:l) {
    wx <- sum(w * xl[i, 1:n])
    margin <- wx * xl[i, n + 1]
    Q <- Q + loss(margin)
  }
  repeat {
    margins <- array(dim = l)
    for (i in 1:l)
    {
      xi <- xl[i, 1:n]
      yi <- xl[i, n + 1]
      margins[i] <- crossprod(w, xi) * yi 
    }
    errorIndexes <- which(margins <= 0) 
    if (length(errorIndexes) > 0)
    {
      i <- sample(1:l, 1)
      iterCount <- iterCount + 1
      xi <- xl[i, 1:n]
      yi <- xl[i, n + 1]
      wx <- crossprod(w, xi)
      margin <- wx * yi
      ex <- loss(margin)
      w <- w + eta * xi * yi * sigmoid(-wx * yi) 
      Qprev <- Q
      Q <- (1 - lambda) * Q + lambda * ex 
    } else
    {
      break
    }
  }
  return(w) 
```

![138](https://github.com/kristinaovc/ML1/blob/master/Regression/log.PNG)

![139](https://github.com/kristinaovc/ML1/blob/master/Regression/log2.PNG)

![140](https://github.com/kristinaovc/ML1/blob/master/Regression/log7.PNG)

![141](https://github.com/kristinaovc/ML1/blob/master/Regression/log6.PNG)

Достоинства логистической регрессии

- Как правило, логистическая регрессия дает лучшие результаты по сравнению с линейным дискриминантом Фишера (поскольку она основана на менее жестких гипотезах), а также по сравнению с дельта-правилом и правилом Хэбба (поскольку она использует „более правильную” функцию потерь).

- Возможность оценивать апостериорные вероятности и риски.

### Все линии

![139](https://github.com/kristinaovc/ML1/blob/master/allLines/allLines.PNG)
