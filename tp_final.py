# coding=utf-8
# 1) Desarrollar un GRASP para el problema de la mochila. Para el mismo se deberán proponer dos heurísticas de búsqueda local distintas y elegir una
# probando cual funciona mejor en el marco del grasp. Los criterios de corte del GRASP deberían estar adaptados a cada una ya que si una produce
# soluciones de mejor calidad pero a mayor costo de tiempo computacional, quizás convenga reducir la cantidad de iteraciones GRASP totales.
#
# 2) Decidir un criterio de subdivisión en problemas (o branching). Ejemplos: incluir o no incluir un determinado ítem en la solución, incluir o no dos
# ítems determinados en la solucion.
#
# 3) De acuerdo al criterio determinado en el punto anterior, dar un algoritmo para elegir concretamente en cada nodo del árbol como dividir el subproblema.
#     Volviendo a los ejemplos del punto anterior, en el primer caso su algoritmo deberá decidir cuál de los ítems aún no usados será el elegido para dividir
# el nodo actual. En el caso del segundo ejemplo, se tratará de elegir entre los pares de ítems para los que aún no se tomó una decisión.
#
# Será necesario que argumenten la opción elegida. En ese sentido, será deseable probar con al menos dos ideas y elegir una en base a resultados computacionales.
#
# 4) Desarrollar una relajación para el problema de la mochila que utilice la información que se tiene en cada nodo acerca de las soluciones allí consideradas.
#
# 5) Desarrollar una "heurística primal", es decir una heurística que, partiendo de la información que se tiene en cada nodo acerca de las solciones allí consideradas,
# genera una solución apta para ese nodo y de la mejor calidad posible en cuanto a la función objetivo.
#
# 6) Utilizar lo desarrollado en los puntos anteriores en un Branch and bound, es decir un algoritmo que, con algun criterio a desarrollar por ustedes (que
# puede ser trivial) tome el proximo nodo a analizar, ejecute para él la heurisitica primal y la relajación, propague esta información a los ancestros y
# genere dos nuevos nodos o bien pode el subarbol actual.
#
# 7) Aplicar lo obtenido sobre un conjunto de instancias que les pasamos adjuntas. Hacer lo mismo sobre el backtracking desarrollado en las practicas y el grasp
# trabajado en el punto 1. Comparar los resultados obtenidos en cuanto a velocidad y efectividad.
#
# Para el 6/7 deberan mostrar informalmente los primeros 3 puntos y para el 11/7 hasta el 6 inclusive y consultar sus ideas para el 7. Los restantes dias serán para
# realizar el punto 7 y presentar correctamente el trabajo realizado.

import sys
import random
import time
import glob
import collections

sys.setrecursionlimit(10 ** 6)


class BackpackProblem:
    @classmethod
    def from_file(cls, file_path):
        with open(file_path, 'r') as f:
            item_amount = int(f.readline())
            items = []
            for n in range(0, item_amount):
                items.append(IntegerItem.from_string(f.readline()))
            pack_max_load = int(f.readline())
            optimun = int(f.readline())
        f.close()
        return BackpackProblem(items, EmptyBackpack(pack_max_load, optimun))

    def __init__(self, items, initial_solution):
        self.all_options = items
        self.initial_solution = initial_solution


class Backpack:
    def __init__(self, pack_max_load, optimun=None):
        self.max_load = pack_max_load
        self.optimun = optimun
        self.load = 0
        self.value = 0

    def add(self, new_item):
        return LoadedBackpack(self, new_item)

    def can_hold(self, item):
        return item.weight <= self.max_load - self.load

    def __repr__(self):
        diff = self.value - self.optimun
        if 0 < diff:
            diff = "+{diff}".format(diff=diff)
        return "Value: {value} (Expected: {optimun}, Diff: {diff}) Load: {load}/{max_load}" \
            .format(value=self.value, optimun=self.optimun, diff=diff, load=self.load, max_load=self.max_load)

    def fitting_items(self, available_items):
        return [item for item in available_items if self.can_hold(item)]

    def has_space(self):
        return self.load < self.max_load

    def is_overloaded(self):
        return self.max_load < self.load


class EmptyBackpack(Backpack):
    def __init__(self, pack_max_load, optimun=None):
        super().__init__(pack_max_load, optimun)
        self.parent = self

    def items(self):
        return []

    def remove(self, item):
        return self


class LoadedBackpack(Backpack):
    def __init__(self, parent, item):
        super().__init__(parent.max_load, parent.optimun)
        self.parent = parent
        self.item = item
        self.load = parent.load + item.weight
        self.value = parent.value + item.value

    def items(self):
        items = self.parent.items()
        items.append(self.item)
        return items

    def remove(self, item):
        if self.item is item:
            return self.parent
        else:
            return self.parent.remove(item).add(self.item)


class Item:
    def __init__(self, value, weight):
        self.value = value
        self.weight = weight


class IntegerItem(Item):
    @classmethod
    def from_string(cls, string):
        identifier, value, weight = string.split()
        return IntegerItem(int(value), int(weight), int(identifier))

    def __init__(self, value, weight, identifier):
        super().__init__(value, weight)
        self.id = identifier

    def __repr__(self):
        return "({id}, {value}, {weight})".format(id=self.id, value=self.value, weight=self.weight)

    def is_fractional(self):
        return False


class FractionalItem(Item):
    def __init__(self, item, fraction):
        super().__init__(item.value * fraction, item.weight * fraction)
        self.source_item = item
        self.fraction = fraction

    def __repr__(self):
        return "({id} * {fraction}, {value}, {weight})" \
            .format(id=self.source_item.id, fraction=self.fraction, value=self.value, weight=self.weight)

    def is_fractional(self):
        return True


class BasicTimer:
    def __init__(self, timeout=None):
        self.begin = 0
        if timeout is None:
            self.timeout = float('inf')
        else:
            self.timeout = timeout

    def start(self):
        self.begin = time.time()

    def run_time(self):
        return time.time() - self.begin

    def is_timeout(self):
        return self.timeout < self.run_time()

    def reset(self):
        self.begin = 0

    def is_running(self):
        return self.begin is not 0


class JointTimer:
    def __init__(self, timer_a, timer_b):
        self.timer_a = timer_a
        self.timer_b = timer_b

    def is_timeout(self):
        return self.timer_a.is_timeout() or self.timer_b.is_timeout()


class Solver:
    def __init__(self, title, problem, timer=None):
        self.title = title
        self.problem = problem
        self.solution = None
        self.total_time = 0
        if timer is None:
            self.internal_timer = BasicTimer()
        else:
            self.internal_timer = timer

    def solve(self, external_timer=None):
        self.internal_timer.start()
        if external_timer is None:
            self.do_solve(self.internal_timer)
        else:
            self.do_solve(JointTimer(self.internal_timer, external_timer))
        self.total_time = self.internal_timer.run_time()

    def report(self):
        text = "{title}:\n\tFinished in {run_time} seconds\n\tSolution: {solution}" \
            .format(title=self.title, run_time=self.total_time, solution=self.solution)
        return text

    def do_solve(self, timer):
        raise NotImplementedError("Please Implement this method")


class BackpackSolver(Solver):
    def do_solve(self, timer):
        raise NotImplementedError("Please Implement this method")

    def report(self):
        base_text = super().report()
        items = self.solution_items()
        extra_text = "\n\tItems: {items}".format(items=items)
        return base_text + extra_text

    def solution_items(self):
        show_items = 5
        if self.solution is None:
            return []
        else:
            items = self.solution.items()
            if show_items < len(items):
                items = items[:show_items]
                items.append("...")
            return items


class GRASPSolver(BackpackSolver):
    def __init__(self, problem, continue_condition, evaluator, greedy_random_solver, refiners, timer=None):
        super().__init__("GRASP", problem, timer)
        self.continue_condition = continue_condition
        self.evaluator = evaluator
        self.greedy_random_solver = greedy_random_solver
        self.refiners = refiners
        self.tries = 0
        self.tries_without_improvement = 0

    def do_solve(self, timer):
        self.solution = self.make_solution(timer)
        self.continue_solving(timer)

    def continue_solving(self, timer):
        while self.continue_condition.should_continue(self) and not timer.is_timeout():
            new_solution = self.make_solution(timer)
            if self.evaluator.better(new_solution, self.solution):
                self.solution = new_solution
                self.tries_without_improvement = 0
            else:
                self.tries_without_improvement += 1

    def make_solution(self, timer):
        self.greedy_random_solver.solve(timer)
        sol = self.greedy_random_solver.solution
        self.tries += 1
        for refiner in self.refiners:
            sol = refiner.refine(sol, timer)
        return sol


class LocalSearchRefiner:
    def __init__(self, evaluator, variators):
        self.evaluator = evaluator
        self.variators = variators


class ExaustiveSequentialLocalSearchRefiner(LocalSearchRefiner):
    def refine(self, starting_solution, timer):
        solution = starting_solution
        for variator in self.variators:
            improved = True
            while improved:
                options = variator.all_variations_of(solution, timer)
                best_option = self.evaluator.best_of_all(options)
                improved = self.evaluator.better(best_option, solution)
                if improved:
                    solution = best_option
        return solution


class RandomSequentialLocalSearchRefiner(LocalSearchRefiner):
    def __init__(self, evaluator, variators, cut_condition, timer=None):
        super().__init__(evaluator, variators)
        self.cut_condition = cut_condition
        if timer is None:
            self.internal_timer = BasicTimer()
        else:
            self.internal_timer = timer
        self.tries = 0
        self.tries_without_improvement = 0

    def reset(self):
        self.tries = 0
        self.tries_without_improvement = 0

    def refine(self, starting_solution, external_timer):
        self.reset()
        self.internal_timer.start()
        timer = JointTimer(self.internal_timer, external_timer)
        best_option = starting_solution
        while self.cut_condition.should_continue(self) and not timer.is_timeout():
            self.tries += 1
            modified = starting_solution
            for variator in self.variators:
                modified = variator.modify(modified, timer)
            if self.evaluator.better(modified, best_option):
                best_option = modified
                self.tries_without_improvement = 0
            else:
                self.tries_without_improvement += 1
        return best_option


class ContinueCondition:
    def logic_and(self, other):
        return ContinueAnd(self, other)

    def logic_or(self, other):
        return ContinueOr(self, other)

    def logic_not(self):
        return ContinueNot(self)


class BinaryContinueCondition(ContinueCondition):
    def __init__(self, cond_a, cond_b):
        self.condA = cond_a
        self.condB = cond_b


class UnaryContinueCondition(ContinueCondition):
    def __init__(self, cond):
        self.cond = cond


class ContinueAnd(BinaryContinueCondition):
    def should_continue(self, code):
        return self.condA.should_continue(code) and self.condB.should_continue(code)


class ContinueOr(BinaryContinueCondition):
    def should_continue(self, code):
        return self.condA.should_continue(code) or self.condB.should_continue(code)


class ContinueNot(UnaryContinueCondition):
    def should_continue(self, code):
        return not self.cond.should_continue(code)


class ContinueByTries(UnaryContinueCondition):
    def should_continue(self, code):
        return code.tries < self.cond


class ContinueByNoImprovement(UnaryContinueCondition):
    def should_continue(self, code):
        return code.tries_without_improvement < self.cond


class MaximizingEvaluator:
    def better(self, sol_a, sol_b):
        return self.cost(sol_a) > self.cost(sol_b)

    def best(self, sol_a, sol_b):
        if self.better(sol_a, sol_b):
            return sol_a
        else:
            return sol_b

    def best_of_all(self, solutions):
        best = max(solutions, key=lambda sol: self.cost(sol))
        return best

    def equal(self, sol_a, sol_b):
        return self.cost(sol_a) == self.cost(sol_b)

    def cost(self, sol):
        raise NotImplementedError("Please Implement this method")


class BackpackEvaluator(MaximizingEvaluator):
    def cost(self, backpack):
        if backpack is None:
            return 0
        else:
            return backpack.value


class BackpackVariator:
    def __init__(self, problem, picker):
        self.problem = problem
        self.picker = picker

    def modify(self, seed_solution, timer, modification_seed=None):
        solution_items = seed_solution.items()
        if modification_seed is None:
            return self.modify(seed_solution, timer, random.choice(solution_items))
        else:
            new_solution = seed_solution.remove(modification_seed)
            allowed_items = [item for item in self.problem.all_options if item not in solution_items]
            new_solution = self.picker.fill(new_solution, allowed_items, timer)
            return new_solution

    def all_variations_of(self, solution, timer):
        return [self.modify(solution, timer, item_for_removal) for item_for_removal in solution.items()]


class BackpackGreedyRandomSolver(BackpackSolver):
    def __init__(self, problem, sorting_key, greediness=1.0, timer=None):
        super().__init__("Greedy", problem, timer)
        self.sorting_key = sorting_key
        self.picker = BackpackGreedyRandomItemPicker(sorting_key, greediness)

    def do_solve(self, timer):
        self.solution = self.picker.fill(self.problem.initial_solution, self.problem.all_options, timer)


class BackpackItemPicker:
    def __init__(self, timer=None):
        if timer is None:
            self.internal_timer = BasicTimer()
        else:
            self.internal_timer = timer

    def fill(self, backpack, available_items, external_timer=None):
        self.internal_timer.start()
        if external_timer is None:
            return self.do_fill(backpack, available_items, self.internal_timer)
        else:
            return self.do_fill(backpack, available_items, JointTimer(self.internal_timer, external_timer))

    def do_fill(self, backpack, available_items, timer):
        raise NotImplementedError("Please Implement this method")


class BackpackGreedyRandomItemPicker(BackpackItemPicker):
    def __init__(self, comparation_key, greedines=1.0, timer=None):
        super().__init__(timer)
        self.comparation_key = comparation_key
        self.greedines = greedines

    def do_fill(self, starting_backpack, available_items, timer):
        item_options = starting_backpack.fitting_items(available_items)
        item_options.sort(key=self.comparation_key, reverse=True)
        backpack = starting_backpack
        while item_options and not timer.is_timeout():
            min_value = self.comparation_key(item_options[-1])
            max_value = self.comparation_key(item_options[0])
            cut_off = max_value - self.greedines * (max_value - min_value)
            filtered_items = [item for item in item_options if cut_off <= self.comparation_key(item)]
            chosen_item = random.choice(filtered_items)
            backpack = backpack.add(chosen_item)
            item_options.remove(chosen_item)
            item_options = backpack.fitting_items(item_options)
        return backpack


class BacktrackingItemPicker(BackpackItemPicker):
    def __init__(self, evaluator, timer=None):
        super().__init__(timer)
        self.evaluator = evaluator

    def do_fill(self, backpack, available_items, timer):
        best_backpack = backpack
        if not timer.is_timeout():
            posible_items = backpack.fitting_items(available_items)
            if posible_items:
                item = posible_items.pop()
                best_pack_with_item = self.do_fill(backpack.add(item), posible_items, timer)
                best_pack_without_item = self.do_fill(backpack, posible_items, timer)
                best_backpack = self.evaluator.best(best_pack_with_item, best_pack_without_item)
        return best_backpack


class BacktrakingSolver(BackpackSolver):
    def __init__(self, problem, recursive_solver, timer=None):
        super().__init__("Backtracking", problem, timer)
        self.recursive_solver = recursive_solver

    def do_solve(self, timer):
        empty_sol = self.problem.initial_solution
        self.solution = self.recursive_solver.fill(empty_sol, self.problem.all_options, timer)


class BranchAndBoundSolver(BackpackSolver):
    def __init__(self, problem, storage, evaluator, timer=None):
        super().__init__("Branch & Bound", problem, timer)
        self.evaluator = evaluator
        self.storage = storage
        self.nodes_visited = 0

    def do_solve(self, timer):
        root_node = self.make_node(self.problem, timer)
        upper_bound = root_node.upper_bound
        lower_bound = root_node.lower_bound
        self.storage.push(root_node)
        self.nodes_visited = 0
        while not (self.storage.is_empty() or timer.is_timeout()):
            node = self.storage.pop()
            lower_bound = self.evaluator.best(lower_bound, node.lower_bound)
            if self.evaluator.equal(node.lower_bound, upper_bound):
                lower_bound = node.lower_bound
                break
            elif self.evaluator.better(node.upper_bound, lower_bound) and \
                    not self.evaluator.better(node.upper_bound, upper_bound):
                branches = self.branch(node.problem)
                new_nodes = [self.make_node(problem, timer) for problem in branches]
                self.storage.push_all(new_nodes)
            # elif self.evaluator.better(lower_bound, node.upper_bound) or \
            #         self.evaluator.equal(lower_bound, node.upper_bound):
            #     pass  # Node is worse tan current best
            # elif self.evaluator.equal(node.lower_bound, node.upper_bound):
            #     pass  # Node branches aren't relevant any more
            self.nodes_visited += 1
        self.solution = lower_bound

    def report(self):
        base = super().report()
        extra = "\n\tTotal nodes visited: {nodes_visited}".format(nodes_visited=self.nodes_visited)
        return base + extra

    def make_node(self, problem, timer):
        upper_bound = self.upper_bound(problem, timer)
        lower_bound = self.lower_bound(problem, timer)
        return BranchAndBoundNode(problem, upper_bound, lower_bound)

    def upper_bound(self, problem, timer):
        raise NotImplementedError("Please Implement this method")

    def lower_bound(self, problem, timer):
        raise NotImplementedError("Please Implement this method")

    def branch(self, problem):
        raise NotImplementedError("Please Implement this method")


class BackpackBnBSolver(BranchAndBoundSolver):
    def __init__(self, problem, storage, evaluator, brancher, timer=None):
        super().__init__(problem, storage, evaluator, timer)
        self.brancher = brancher

    def upper_bound(self, problem, timer):
        solver = BackpackFractionalSolver(
            problem,
            lambda item: item.value / item.weight)
        solver.solve(timer)
        return solver.solution

    def lower_bound(self, problem, timer):
        solver = BackpackGreedyRandomSolver(
            problem,
            lambda item: item.value / item.weight,
            0)
        improver = BackpackPrimalHeuristic(
            problem,
            BacktrackingItemPicker(self.evaluator),
            BasicTimer(3)
        )
        solver.solve(timer)
        solution = solver.solution
        solution = improver.improve(solution)
        return solution

    def branch(self, problem):
        return self.brancher.branch(problem)


class BackpackYesNoBrancher:
    def __init__(self, item_selector):
        self.selector = item_selector

    def branch(self, problem):
        branches = []
        usable_items = [item for item in problem.all_options if problem.initial_solution.can_hold(item)]
        if usable_items:
            branching_item = self.selector.pick_from(usable_items)
            usable_items.remove(branching_item)
            branches.append(BackpackProblem(usable_items, problem.initial_solution.add(branching_item)))
            branches.append(BackpackProblem(usable_items, problem.initial_solution))
        return branches


class Selector:
    def __init__(self, key):
        self.key = key

    def pick_from(self, options):
        raise NotImplementedError("Please Implement this method")


class MaxSelector(Selector):
    def pick_from(self, options):
        return max(options, key=self.key)


class MinSelector(Selector):
    def pick_from(self, options):
        return min(options, key=self.key)


class BranchAndBoundNode:
    def __init__(self, problem, upper, lower):
        self.problem = problem
        self.upper_bound = upper
        self.lower_bound = lower


class PrimalHeuristic:
    def __init__(self, timer=None):
        if timer is None:
            self.internal_timer = BasicTimer()
        else:
            self.internal_timer = timer

    def improve(self, bound, external_timer=None):
        self.internal_timer.start()
        if external_timer is None:
            return self.do_improve(bound, self.internal_timer)
        else:
            return self.do_improve(bound, JointTimer(self.internal_timer, external_timer))

    def do_improve(self, bound, timer):
        raise NotImplementedError("Please Implement this method")


class BackpackPrimalHeuristic(PrimalHeuristic):
    def __init__(self, problem, picker, timer=None):
        super().__init__(timer)
        self.problem = problem
        self.picker = picker

    def do_improve(self, bound, timer):
        base_solution = bound.parent
        pack_items = base_solution.items()
        usable_items = [item for item in self.problem.all_options if item not in pack_items]
        improved = self.picker.fill(base_solution, usable_items, timer)
        return improved


class Storage:
    def __init__(self):
        self.data = collections.deque()

    def pop(self):
        raise NotImplementedError("Please Implement this method")

    def push(self, elem):
        raise NotImplementedError("Please Implement this method")

    def push_all(self, elems):
        for elem in elems:
            self.push(elem)

    def is_empty(self):
        return len(self.data) is 0


class FiFoStorage(Storage):
    def pop(self):
        return self.data.popleft()

    def push(self, elem):
        self.data.append(elem)


class LiFoStorage(Storage):
    def pop(self):
        return self.data.pop()

    def push(self, elem):
        self.data.append(elem)


class BackpackFractionalSolver(BackpackGreedyRandomSolver):
    def __init__(self, problem, sorting_key, timer=None):
        super().__init__(problem, sorting_key, 0, timer)

    def do_solve(self, timer):
        super().do_solve(timer)
        pack_items = self.solution.items()
        leftover_items = [item for item in self.problem.all_options if item not in pack_items]
        leftover_items.sort(key=self.sorting_key, reverse=True)
        if self.solution.has_space() and leftover_items:
            item = leftover_items.pop(0)
            free_load = self.solution.max_load - self.solution.load
            fractional_item = FractionalItem(item, free_load / item.weight)
            self.solution = self.solution.add(fractional_item)


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


def do_branch_and_bound1(backpack_problem):
    solver = BackpackBnBSolver(
        backpack_problem,
        LiFoStorage(),
        BackpackEvaluator(),
        BackpackYesNoBrancher(MinSelector(key=lambda item: item.value / item.weight)),
        BasicTimer(300))
    run(solver)


def do_branch_and_bound2(backpack_problem):
    solver = BackpackBnBSolver(
        backpack_problem,
        FiFoStorage(),
        BackpackEvaluator(),
        BackpackYesNoBrancher(MaxSelector(key=lambda item: item.value / item.weight)),
        BasicTimer(300))
    run(solver)


def do_backtracking(backpack_problem):
    solver = BacktrakingSolver(
        backpack_problem,
        BacktrackingItemPicker(BackpackEvaluator()),
        BasicTimer(300))
    run(solver)


def do_greedy(backpack_problem):
    solver = BackpackGreedyRandomSolver(
        backpack_problem,
        lambda item: item.value / item.weight,
        0,
        BasicTimer(300))
    run(solver)


def do_fractional(backpack_problem):
    solver = BackpackFractionalSolver(
        backpack_problem,
        lambda item: item.value / item.weight,
        BasicTimer(300))
    run(solver)


def run(solver):
    solver.solve()
    print(solver.report())


def run_test_file(file_path):
    print("**************** Running file {file} ****************".format(file=file_path))
    for _ in range(0,3):
        problem = BackpackProblem.from_file(file_path)
        do_grasp1(problem)
        do_grasp2(problem)
        do_branch_and_bound1(problem)
        do_branch_and_bound2(problem)
        do_backtracking(problem)
        # do_greedy(problem)
        # do_fractional(problem)
        print("**************************************************")


def run_all_tests():
    files = glob.glob('tests/*.in')
    for file_ in files:
        run_test_file(file_)


def main():
    # run_test_file('tests/test_012_2e1.in')
    # run_test_file('tests/test_013_2e1.in')
    # run_test_file('tests/test_014_2e1.in')
    # run_test_file('tests/test_015_2e2.in')
    # run_test_file('tests/test_016_2e2.in')
    # run_test_file('tests/test_017_2e2.in')
    # run_test_file('tests/test_018_1e3.in')
    # run_test_file('tests/test_019_1e3.in')
    # run_test_file('tests/test_020_1e3.in')
    # run_test_file('tests/test_021_2e3.in')
    # run_test_file('tests/test_022_2e3.in')
    # run_test_file('tests/test_023_2e3.in')
    run_all_tests()


if __name__ == '__main__':
    main()
