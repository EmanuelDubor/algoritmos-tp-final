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
esta compuesto por la lista de items disponibles y la mochila a llenar.

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

La implementacion generica de Branch and Bound se encuentra en la clase `BranchAndBoundSolver` y
provee de tres metodos que el usuario debe sobreescribir a fin de alcanzar un objeto concreto capaz
de resolver el problema en cuestion.

#### Backtracking






