import numpy as np
import random
import math
from typing import *
from entities import *
from functools import reduce


def euclidean_distance(coordinate: Coordinate, center: Center) -> float:
    return np.sqrt(np.sum(np.power(np.array(coordinate) - np.array(center), 2)))


def ecost_of_an_assignment(u_points: List[UncertainPoint], centers: List[Center],
                           assignments: Assignments) -> float:

    objects = [{'i': i, 'prob': point.prob,
                'distance_to_center': euclidean_distance(point.coordinate, centers[assignments[i]])}
                for (i, up) in enumerate(u_points)
                for point in up]

    objects.sort(key=lambda x: x.get('distance_to_center'))

    sum_probs = [0] * len(u_points)
    number_of_zero_sum_probs = len(u_points)
    mult, ecost = 1, 0

    for obj in objects:
        i, prob, dist = obj.get('i'), obj.get('prob'), obj.get('distance_to_center')

        if is_equal_to_zero(prob):
            continue

        if is_equal_to_zero(sum_probs[i]):
            number_of_zero_sum_probs -= 1

        if not is_equal_to_zero(sum_probs[i]):
            mult /= sum_probs[i]

        sum_probs[i] += prob
        mult *= sum_probs[i]

        if number_of_zero_sum_probs > 0:
            continue

        ecost += prob * (mult / sum_probs[i]) * dist

    return ecost


def generate_k_center_with_k_center(coordinates: List[Coordinate], k: int) -> List[Center]:
    # start_index = random.randint(0, len(coordinates) - 1)
    start_index = 0

    centers = [coordinates[start_index]]

    while len(centers) < k:
        dists = list(map(lambda coord: get_distance_to_nearest_center(coord, centers), coordinates))
        index = int(np.argmax(dists))   
        centers.append(coordinates[index])

    return centers



def get_expected_coordinate_of_uncertain_point(up:UncertainPoint) -> Coordinate:
    return list(np.dot([p.prob for p in up], [p.coordinate for p in up]))
    
    
def get_index_of_nearest_center_to_coordinate(coordinate: Coordinate, centers: List[Center]) -> Center:
    return int(np.argmin(list(map(lambda c: euclidean_distance(coordinate, c), centers))))


def get_distance_to_nearest_center(coordinate: Coordinate, centers: List[Center]) -> float:
    return min([euclidean_distance(coordinate, c) for c in centers])
    

def get_expected_distance_of_uncertain_point_to_center(up: UncertainPoint, center: Center) -> float:
    return np.sum(np.dot([p.prob for p in up], [euclidean_distance(p.coordinate, center) for p in up]))


def get_expected_distance_assignment_for_one_uncertain_point(u_point:UncertainPoint, centers:List[Center]) -> int:
    return int(np.argmin(list(map(lambda c: get_expected_distance_of_uncertain_point_to_center(u_point, c), centers))))


def get_centers_of_districts(mins: List[float], maxs: List[float], num_of_intervals: List[int]) -> List[Center]:
    interval_lengths = np.divide(np.array(maxs) - np.array(mins), num_of_intervals)
    permutations = get_all_permutations(len(interval_lengths), num_of_intervals)
    return list(map(lambda per: [mins[i] + (p+0.5)*interval_lengths[i]
                                        for (i, p) in enumerate(per)], permutations))


def get_one_center_of_uncertain_point(up: UncertainPoint, num_of_intervals=10) -> Center:
    mins = np.min(list(map(lambda p: p.coordinate, up)), axis=0)
    maxs = np.max(list(map(lambda p: p.coordinate, up)), axis=0)
    one_centers = get_centers_of_districts(mins, maxs, [num_of_intervals] * len(mins))
    return one_centers[get_expected_distance_assignment_for_one_uncertain_point(up, one_centers)]


def get_all_complete_assignments(temp_assignment: Assignments, k:int) -> List[Assignments]:
    def get_assignments_for_one_permutation(per: List[int], h_indices: List[int]):
        np_assignments = np.array(temp_assignment)
        np_assignments[h_indices] = per
        return list(np_assignments)

    hole_indices = list(filter(lambda i: temp_assignment[i] == -1, range(len(temp_assignment))))
    permutations = get_all_permutations(len(hole_indices), [k] * len(hole_indices))
    return list(map(lambda per:get_assignments_for_one_permutation(per, hole_indices), permutations))    


def get_best_assignments_in_list(u_points: List[UncertainPoint], centers: List[Center],
                                 assignments_list: List[Assignments], spark) -> Assignments:

    assignments_list_par = spark.sparkContext.parallelize(assignments_list, numSlices=250)
    index = np.argmin(assignments_list_par.map(lambda a: ecost_of_an_assignment(u_points, centers, a)).collect())
    return assignments_list[index]

def get_all_permutations(n: int, k_lengths: List[int]) -> List[List[int]]:
    number_of_permutations = reduce(lambda acc, curr: acc * curr, k_lengths)
    curr_per = np.array([0] * n)
    permutations = [list(curr_per)]

    while len(permutations) < number_of_permutations:
        curr_index = 0
        while curr_per[curr_index] == k_lengths[curr_index] - 1:
            curr_per[curr_index] = 0
            curr_index += 1
        curr_per[curr_index] += 1
        permutations.append(list(curr_per))

    return permutations

def is_equal_to_zero(number: int):
    return abs(number) < 1e-6

def check_params_is_valid_for_bagging_assignments(n: int, z: int, k: int, b: int):
    o = (n / b) * math.pow(k, b) * n * z * math.log(n * z, 2)
    return o < 5e11
