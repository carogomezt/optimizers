"""
This implementation for tabu search is modified from:
https://github.com/mohanadkaleia/blog/blob/master/content/posts/tabu-search-gentle-introduction.md
https://github.com/mohanadkaleia/blog/blob/master/content/posts/tabu-search-gentle-introduction.md

Reference:
https://www.researchgate.net/publication/242527226_Tabu_Search_A_Tutorial
"""
import argparse
import copy
import math
import random

import numpy as np
from matplotlib import pyplot as plt
from time import time


def distance(point1, point2):
    """ This function calculates the distance between two points.

        Parameters
        ----------
        point1: dict
            First point.
        point2: dict
            Second point.

        Returns
        -------
        float
            Distance of the two points.
    """
    return math.sqrt((point1['x'] - point2['x']) ** 2 + (point1['y'] - point2['y']) ** 2)


def generate_neighbors(points):
    """ This function generates a 2D distance matrix between all points

        Parameters
        ----------
        points: list
            Coordinates x and y of the initial points.

        Returns
        -------
        dict_of_neighbors: dict
            Dictionary with the points and their distance to other points.
    """
    dict_of_neighbors = {}

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if i not in dict_of_neighbors:
                dict_of_neighbors[i] = {}
                dict_of_neighbors[i][j] = distance(points[i], points[j])
            else:
                dict_of_neighbors[i][j] = distance(points[i], points[j])
            if j not in dict_of_neighbors:
                dict_of_neighbors[j] = {}
                dict_of_neighbors[j][i] = distance(points[j], points[i])
            else:
                dict_of_neighbors[j][i] = distance(points[j], points[i])

    return dict_of_neighbors


def generate_first_solution(nodes, dict_of_neighbors):
    """ This function generates the first solution evaluating each point and adding them to the
        solution if the y have the minimum distance.

        Parameters
        ----------
        nodes: list
            Coordinates x and y of the initial points.
        dict_of_neighbors : dict
            Dictionary with the neighbors of the firs solution.


        Returns
        -------
        first_solution: list
            List with the cities of the first solution.
        distance: float
            Distance of the first solution.
    """
    start_node = nodes[0]
    end_node = start_node

    first_solution = []
    distance = 0
    visiting = start_node
    while visiting not in first_solution:
        neighbors = copy.deepcopy(dict_of_neighbors[visiting])
        neighbors = [i for i in neighbors.items() if i[0] not in first_solution]
        if neighbors:
            next_node = min(neighbors, key=lambda x: x[1])[0]
            distance += dict_of_neighbors[visiting][next_node]
        else:
            distance += dict_of_neighbors[visiting][end_node]
        first_solution.append(visiting)
        visiting = next_node

    first_solution.append(end_node)
    return first_solution, distance


def find_neighborhood(solution, dict_of_neighbors, n_opt=1):
    """ This function find the neighborhood of a solution.

        Parameters
        ----------
        solution: list
            Coordinates x and y of the initial points.
        dict_of_neighbors : dict
            Dictionary with the neighbors of the firs solution.
        n_opt: int
            Number of points to swap.


        Returns
        -------
        neighborhood_of_solution: list
            List with the neighbors of a solution.
    """
    neighborhood_of_solution = []
    limit = len(solution) // 2
    for i in range(1, limit + 1):
        idx1 = []
        n = random.randint(1, len(solution) - 2)
        n_index = solution.index(n)
        for i in range(n_opt):
            idx1.append(n_index + i)

        for j in range(1, limit + 1):
            idx2 = []
            kn = random.randint(1, len(solution) - 2)
            kn_index = solution.index(kn)
            for i in range(n_opt):
                idx2.append(kn_index + i)
            if bool(
                    set(solution[idx1[0]:(idx1[-1] + 1)]) &
                    set(solution[idx2[0]:(idx2[-1] + 1)])):
                continue

            _tmp = copy.deepcopy(solution)
            for i in range(n_opt):
                _tmp[idx1[i]] = solution[idx2[i]]
                _tmp[idx2[i]] = solution[idx1[i]]

            distance = 0
            for k in _tmp[:-1]:
                next_node = _tmp[_tmp.index(k) + 1]
                distance = distance + dict_of_neighbors[k][next_node]
            _tmp.append(distance)
            if _tmp not in neighborhood_of_solution:
                neighborhood_of_solution.append(_tmp)

    indexOfLastItemInTheList = len(neighborhood_of_solution[0]) - 1

    neighborhood_of_solution.sort(key=lambda x: x[indexOfLastItemInTheList])
    return neighborhood_of_solution


def tabu_search(first_solution, distance_of_first_solution, dict_of_neighbors, iters, size,
                n_opt=1):
    """ This function apply the Tabu Search.

        Parameters
        ----------
        first_solution: list
            Coordinates x and y of the initial points.
        distance_of_first_solution: float
            Distance of the first solution.
        dict_of_neighbors : dict
            Dictionary with the neighbors of the first solution.
        iters: int
            Max number of iterations
        size: int
            Tabu Tenure
        n_opt: int
            Number of points to swap.


        Returns
        -------
        best_solution_ever: list
            List with the cities of the best solution.
        best_cost: float
            Distance of the best solution.
        data: dict
            Information of each iteration and its best distance.
    """
    count = 1
    solution = first_solution
    tabu_list = list()
    past_cost = distance_of_first_solution
    best_cost = distance_of_first_solution
    best_solution_ever = solution
    frequency = {}
    long_term_iter = 0
    diversify = True
    intensify = True
    data = []
    while count <= iters:
        data.append({'iter': count, 'cost': best_cost}, )

        # print(f'Iter: {count}, Cost: {best_cost}')

        if (past_cost - best_cost) <= 0:
            long_term_iter += 1
        else:
            long_term_iter = 0

        # Long term memory: Diversifying and Intensifying the search.
        if long_term_iter >= 50 and frequency:
            new_freq = {k: v for k, v in sorted(frequency.items(), key=lambda item: int(item[1]))}
            exchange_params = []
            if diversify:
                # Diversification phase
                # print('----- Diversifying the Solution -----')
                first_key = list(new_freq.keys())[0]
                exchange_params = list(first_key.split(','))
                frequency.pop(first_key)
                diversify = False
                intensify = True
            elif intensify:
                # Intensification phase
                # print('----- Intensifying the Solution -----')
                last_key = list(new_freq.keys())[-1]
                exchange_params = list(last_key.split(','))
                frequency.pop(last_key)
                diversify = True
                intensify = False
            first_index = int(exchange_params[0])
            second_index = int(exchange_params[1])
            index_first_exchange = solution.index(first_index)
            index_second_exchange = solution.index(second_index)
            solution[index_first_exchange] = second_index
            solution[index_second_exchange] = first_index

        elif long_term_iter >= 50 and not frequency:
            break

        neighborhood = find_neighborhood(solution, dict_of_neighbors, n_opt=n_opt)
        # print("Neighbors: ", len(neighborhood))

        index_of_best_solution = 0
        best_solution = neighborhood[index_of_best_solution]
        best_cost_index = len(best_solution) - 1

        past_cost = best_cost
        found = False
        while found is False:
            first_exchange_node, second_exchange_node = [], []
            n_opt_counter = 0
            for i in range(len(best_solution)):
                if best_solution[i] != solution[i]:
                    first_exchange_node.append(best_solution[i])
                    second_exchange_node.append(solution[i])
                    n_opt_counter += 1
                    if n_opt_counter == n_opt:
                        break

            exchange = first_exchange_node + second_exchange_node
            cost = neighborhood[index_of_best_solution][best_cost_index]
            # Aspiration criteria: if the solution is better then take it no matter if is tabu.
            if cost < best_cost:
                tabu_list.append(exchange)
                best_cost = cost
                solution = best_solution[:-1]
                best_solution_ever = solution
                found = True
                s = [str(i) for i in exchange]
                freq_param = ','.join(s)
                if freq_param in frequency:
                    frequency[freq_param] += 1
                else:
                    frequency[freq_param] = 1
            # Short term memory
            if first_exchange_node + second_exchange_node not in tabu_list and \
                    second_exchange_node + first_exchange_node not in tabu_list:
                tabu_list.append(exchange)
                found = True
                solution = best_solution[:-1]
                cost = neighborhood[index_of_best_solution][best_cost_index]
                if cost < best_cost:
                    best_cost = cost
                    best_solution_ever = solution
                    s = [str(i) for i in exchange]
                    freq_param = ','.join(s)
                    if freq_param in frequency:
                        frequency[freq_param] += 1
                    else:
                        frequency[freq_param] = 1
            elif index_of_best_solution < len(neighborhood):
                best_solution = neighborhood[index_of_best_solution]
                index_of_best_solution = index_of_best_solution + 1

        while len(tabu_list) > size:
            tabu_list.pop(0)

        count = count + 1

    print(f'Iter: {count}, Cost: {best_cost}')
    best_solution_ever.pop(-1)
    return best_solution_ever, best_cost, data


def process_file_input(file_name):
    """ This function format the points in a file.

        Parameters
        ----------
        file_name: str
            Name of the file with the points.

        Returns
        -------
        points: list
            List with the coordinates of the cities.
    """
    with open(file_name) as f:
        lines = f.readlines()
    index = lines.index('NODE_COORD_SECTION\n')
    coordinates = lines[index + 1: -1]
    points = []
    for coordinate in coordinates:
        coordinate = coordinate.replace('\n', '')
        _, x, y = coordinate.split()
        points.append({'x': float(x), 'y': float(y)})

    return points


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tabu Search")
    parser.add_argument(
        "-f",
        "--File",
        type=str,
        help="Path to the file containing the data",
        required=True,
    )
    parser.add_argument(
        "-i",
        "--Iterations",
        type=int,
        help="How many iterations the algorithm should perform",
        required=True,
    )
    args = parser.parse_args()

    input_file = args.File
    input_name = input_file.split('.')[0]
    points = process_file_input(input_file)
    x_points = np.array([p['x'] for p in points])
    y_points = np.array([p['y'] for p in points])

    # Graph the points
    # plt.scatter(x_points, y_points)
    # for i, txt in enumerate(nodes):
    #     plt.annotate(txt, (x_points[i], y_points[i]))
    # plt.show()

    # print(points)
    try:
        with open(f'{input_name}_neighborhood.txt') as f:
            lines = f.readlines()
            initial_neighbors = eval(lines[0])
            # print(initial_neighbors)
    except IOError:
        start_time = time()
        print(f'Generating Neighbors')
        initial_neighbors = generate_neighbors(points)
        print(initial_neighbors)
        elapsed_time = time() - start_time
        print(f'Finished the generation of the neighborhood and it took: {elapsed_time}s')
        neighborhood_file = open(f'{input_name}_neighborhood.txt', "w")
        neighborhood_file.write(f'{initial_neighbors}')
        neighborhood_file.close()

    nodes = list(range(len(points)))
    try:
        with open(f'{input_name}_first_solution.txt') as f:
            lines = f.readlines()
            f_solution = lines[0].replace('\n', '')
            first_solution = list(f_solution.split(','))
            first_solution = [int(i) for i in first_solution]
            distance_of_first_solution = float(lines[1])
            # print(first_solution, distance_of_first_solution)
    except IOError:
        start_time = time()
        print(f'Generating First Solution')
        first_solution, distance_of_first_solution = generate_first_solution(nodes,
                                                                             initial_neighbors)
        print(first_solution, distance_of_first_solution)
        elapsed_time = time() - start_time
        print(f'Finished the generation of the First Solution and it took: {elapsed_time}s')
        s = [str(i) for i in first_solution]
        fist_solution_casted = ','.join(s)
        first_solution_file = open(f'{input_name}_first_solution.txt', "w")
        first_solution_file.write(f'{fist_solution_casted}\n{distance_of_first_solution}')
        first_solution_file.close()

    # Change this value to run the code multiple times.
    n = 1
    for i in range(n):
        print('\n')
        iters = args.Iterations
        size = math.ceil(math.sqrt(len(points)))
        start_time = time()
        print('\n')
        print(f'Tabu Search N°: {i}')
        solution, dist, data = tabu_search(first_solution, distance_of_first_solution,
                                           initial_neighbors, iters, size)
        print(solution, dist)
        elapsed_time = time() - start_time
        print(f'Finished Tabu Search and it took: {elapsed_time}s')
        optimum = 9352  # QA194 - Qatar - 194 Cities
        error = abs(dist - optimum)
        print("Error: ", error)
        print("Accuracy: ", (dist * 100) / optimum)
        iterations = np.array([p['iter'] for p in data])
        costs = np.array([p['cost'] for p in data])
        # norm = np.linalg.norm(costs)
        # normal_cost = costs/norm
        # plt.scatter(iterations, normal_cost)

        # Graph the cost against the number of iterations
        plt.plot(iterations, costs)
        plt.ylabel('Cost')
        plt.xlabel('Iterations')
        # for i, txt in enumerate(nodes):
        #     plt.annotate(txt, (x_points[i], y_points[i]))
        plt.savefig(f'{input_name}-cost-vs-iters.png')
