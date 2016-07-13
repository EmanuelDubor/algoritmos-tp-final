# Algoritmos - Trabajo practico final

## Objetivo

El objetivo de este trabajo consiste en implementar y probar empiricamente la efectividad de disitintas
versiones de algoritmos capaces de solucionar el conocido *problema de la mochila entera*.
Si bien el trabajo no pretende hacer un estudio exaustivo de las distintas tecnicas empleadas si es
necesario que se justifiquen las decisiones de implementacion tomadas en los puntos relevantes, al menos
de manera informal.

## Implementacion

#### Solver

La abstraccion generica de *solver* pretende abarcar a todo objeto capaz de generar una solucion a
partir de un problema inicial en un tiempo acotado o no. En el caso de este trabajo, dicho problema
esta compuesto por la lista de items disponibles y la mochila a completar con dichos items.

#### G.R.A.S.P

El algoritmo GRASP, implementado en la clase `GRASPSolver`, esta diseñado para ser flexible. Al momento
de instanciar el objeto es necesario que el usuario provea una serie de sub-componentes que son los
que determinan el resultado final de la ejecucion.

Los sub-componentes mas relevantes son:
 * `continue_condition`: Este objeto (o composicion de objetos) representa la condicion de  corte del
 algoritmo. Los criterios atomicos soportados son `ContinueByTries` y `ContinueByNoImprovement` asi
 como cualquier expresion logica que los combine.
 * `greedy_random_solver`: Este objeto es el encargado de proveer las soluciones iniciales sobre las
 que trabajara el primero de los *refiners*. Como el nombre lo indica, se espera que provea soluciones
 greedy pseudo-aleatorias pero en realidad puede utilizarse cualquier clase de *solver*.
 * `refiners`: En este caso se espera una lista de objetos en lugar de solo uno. Cada uno de estos
 sera aplicado de forma secuencial al resultado del anterior y son los que determinan la vecindad de
 una solucion asi como tambien la manera de explorarla.

#### Branch and Bound

La implementacion generica de *Branch and Bound* se encuentra en la clase `BranchAndBoundSolver` y
provee de tres metodos que el usuario debe sobreescribir a fin de alcanzar un objeto concreto capaz
de resolver el problema en cuestion. Estos tres metodos (`upper_bound`, `lower_bound` y `branch`)
junto con los subcomponentes que el algoritmo recibe al momento de instanciarse son los que determinan
el orden en el que se genera y explora el el arbol de soluciones.

#### Backtracking

El algoritmo de backtracking, al ser comparativamente sencillo resulto mucho mas facil de implementar;
el `BacktrackingSolver` simplemente entrega la solucion inicial del problema (en este caso, la mochila
vacia) a un sub-componente encargado de ir llenandola de manera recursiva.

## Implementaciones concretas de los algoritmos

#### Consideraciones generales

1. Todos los algoritmos disponen de un tiempo de ejecucion maximo de  aproximadamente 300
    segundos (5 minutos).
2. En ambos algoritmos GRASP se utiliza la misma vecindad, definida mediante el siguiente proceso:
    1. Tomar la mochila inicial y quitarle un item.
    2. Completar, mediante backtracking, el espacio libre que queda.

    Este tipo de vecindad hace que las soluciones generadas sean potencialmente mejores que la
    inicial ya que parte del espacio de la mochila esta utilizado de forma optima. Adicionalmente,
    no van a destinarse mas de 3 segundos al proceso de backtracking.
3. En ambos algoritmos *Branch and Bound* las cotas inferiores y superiores se calculan de la
    siguiente forma:
    * Cota superior: Relajacion de mochila fraccionaria.
    * Cota inferior: Mochila *greedy* mejorada mediante backtracking de no mas de 3 segundos.
        Cabe destacar que el proceso de backtracking opera sobre el espacio libre que queda luego
        de sacar de la mochila el ultimo item agregado por la solucion *greedy*. Esto provee una
        potencial mejora sobre la mochila *greedy* ya que optimiza parte del espacio de la misma.

#### GRASP 1

    def do_grasp1(backpack_problem):
        solver = GRASPSolver(
            backpack_problem,
            ContinueByTries(100).logic_and(ContinueByNoImprovement(5)),
            BackpackEvaluator(),
            BackpackGreedyRandomSolver(backpack_problem, sorting_key=lambda item: item.value / item.weight, greediness=0.3),
            [
                ExaustiveSequentialLocalSearchRefiner(
                    BackpackEvaluator(),
                    [
                        BackpackVariator(backpack_problem, BacktrackingItemPicker(BackpackEvaluator(), BasicTimer(3)))
                    ]
                )
            ],
            BasicTimer(300))
        run(solver)

Este algoritmo concreto parte de soluciones *greedy* aleatorias generadas segun la tasa valor/peso
de cada item, con un factor de *greedines* de 30%; es decir, cualquier item cuya tasa este a un 30%
o menos de la mejor tasa posible puede ser incorporado a la solucion que se esta construyendo.

Una vez generada la solucion basica el algoritmo aplica sobre ella el unico *refiner* que tiene,
haciendo una busqueda local exaustiva dentro de la vecindad. Luego de analizar todas las soluciones
vecinas generadas, el algoritmo toma la mejor, genera una nueva vecindad y continua el proceso hasta
hallar una solucion en la que ningun vecino es mejor que la solucion que ya se tiene.

Finalmente el algoritmo finaliza luego de haber generado y refinado 100 soluciones iniciales o de
haber generado y refinado 5 soluciones iniciales sin obtener una mejor que la ya disponible.
Esta condicion de corte se eligio tomando en cuenta que la busqueda local es exaustiva dentro de la
vecindad de la solucion inicial; lo que, segun el criterio de vecindad anteriormente definido, puede
llevar a generar y analizar muchisimas soluciones.

#### GRASP 2

    def do_grasp2(backpack_problem):
        solver = GRASPSolver(
            backpack_problem,
            ContinueByTries(300).logic_and(ContinueByNoImprovement(30)),
            BackpackEvaluator(),
            BackpackGreedyRandomSolver(backpack_problem, sorting_key=lambda item: item.value / item.weight, greediness=0.3),
            [
                RandomSequentialLocalSearchRefiner(
                    BackpackEvaluator(),
                    [
                        BackpackVariator(backpack_problem, BacktrackingItemPicker(BackpackEvaluator(), BasicTimer(3)))
                    ],
                    ContinueByTries(5).logic_and(ContinueByNoImprovement(3)),
                    BasicTimer(10))
            ],
            BasicTimer(300))
        run(solver)

Igual que el anterior, este algoritmo parte de soluciones *greedy* aleatorias generadas segun la
tasa valor/peso del item y con un factor de *greedines* del 30%; la diferencia radica en como se
realiza la busqueda local y bajo que condiciones el algoritmo se detiene.

En este caso, la busqueda local es aleatoria. El *refiner* dispone de hasta 10 segundos en los que
genera aleatoriamente un vecino de la solucion inicial y analiza si es mejor que la solucion
inicial, repitiendo el proceso hasta agotar el tiempo, probar 5 vecinos o probar 3 vecinos sin
obtener mejoras.

Como condicion de corte de este algoritmo se dan 300 iteraciones o 30 sin mejoras ya que la busqueda
local es mas rapida pero menos fiable dado que su naturaleza aleatoria le permite explorar varias
veces el mismo vecino.

#### Branch and Bound 1

    def do_branch_and_bound1(backpack_problem):
        solver = BackpackBnBSolver(
            backpack_problem,
            LiFoStorage(),
            BackpackEvaluator(),
            BackpackYesNoBrancher(MinSelector(key=lambda item: item.value / item.weight)),
            BasicTimer(300))
        run(solver)

Esta implementacion de *branch and bound* utiliza una pila para determinar que nodo del arbol es el
proximo a analizar. La forma de dividir un nodo es mediante poner o no poner un item en la solucion
mientras que es responsabilidad del *selector* elegir que item sera utilizado para dividir el nodo.

En esta configuracion, se eligen primero los items cuya tasa de valor/sobre peso es la **peor** de
todas. La razon detras de esta decision es que un item con la peor tasa probablemente lleve a
soluciones muy malas ya que ocupa mucho espacio y no mejora significativamente el valor de la
mochila. A raiz de esto, cuanto antes se pueda descartar el item como parte de la solucion optima
menos nodos tendran que explorarse.

#### Branch and Bound 2

    def do_branch_and_bound2(backpack_problem):
        solver = BackpackBnBSolver(
            backpack_problem,
            FiFoStorage(),
            BackpackEvaluator(),
            BackpackYesNoBrancher(MaxSelector(key=lambda item: item.value / item.weight)),
            BasicTimer(300))
        run(solver)

Esta implementacion es una leve variacion de la configuracion anterior; en este caso se utiliza una
cola de nodos a analizar y se dividen un nodo en base al item con la **mejor** tasa valor/peso,
alegando que ese item probablemente sea parte de la solucion optima y por lo tanto es mejor
considerar las consecuencias de llevarlo o no lo antes posible.

## Resultados

[Ver planilla de calculo](https://drive.google.com/open?id=1Strh121v9gK7DO-FMkp2ZQhU-dRjScLAlljs-0kmn_U)

## Conclusiones

Puede concluirse que el algoritmo GRASP en su segunda version (vecindad aleatoria) es el mas
eficiente de todos los considerados siendo que es el unico que logro encontrar soluciones optimas
(o mas optimas que los demas algoritmos) de manera consistente y sin utilizar todo el tiempo de
ejecucion disponible, incluso para las instancias mas grandes.

En el caso de los algoritmos *Branch and Bound* puede observarse que ambos algoritmos se comportan
de forma similar, tanto en calidad de soluciones generadas como en efectividad del criterio de
division de los nodos.




