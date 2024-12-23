from typing import *
from entities import *
from utils import *
import numpy as np
import time
from functools import reduce


def get_optimal_assignments(u_points: List[UncertainPoint], centers: List[Center]) -> Assignments:
    all_assignments = get_all_permutations(len(u_points), len(centers))
    return get_best_assignments_in_list(u_points, centers, all_assignments)


def get_expected_distance_assignments(u_points: RDD, centers: List[Center]) -> Assignments:
    return u_points.map(lambda up: get_expected_distance_assignment_for_one_uncertain_point(up, centers)).collect()


def get_expected_point_assignments(u_points: RDD, centers: List[Center]) -> Assignments:
    def get_expected_point_assignment_for_one_uncertain_point(up: UncertainPoint) -> int:
        expected_coordinate = get_expected_coordinate_of_uncertain_point(up)
        return int(np.argmin(list(map(lambda c: euclidean_distance(expected_coordinate, c), centers))))

    return u_points.map(get_expected_point_assignment_for_one_uncertain_point).collect()
    # return list(map(lambda up: get_expected_point_assignment_for_one_uncertain_point(up), u_points))  


def get_one_center_assignments(u_points: List[UncertainPoint], centers:List[Center]) -> Assignments:
    def get_one_center_assignment_for_one_uncertain_point(up:UncertainPoint) -> int:
        one_center = get_one_center_of_uncertain_point(up)
        return int(np.argmin(list(map(lambda c: euclidean_distance(one_center, c), centers))))

    return list(map(lambda up: get_one_center_assignment_for_one_uncertain_point(up), u_points))


def get_probable_center_assignment(u_points: RDD, centers: List[Center]) -> Assignments:
    def get_probable_center_assignment_for_one_point(up: UncertainPoint) -> int:
        def update_accumulator(acc: List[float], curr: Point) -> List[float]:
            dist_from_centers = list(map(lambda c: euclidean_distance(curr.coordinate, c), centers))
            closest_center_index = int(np.argmin(dist_from_centers))
            acc[closest_center_index] += curr.prob
            return acc

        center_probs = reduce(update_accumulator, up, [0] * len(centers))
        expected_center = np.dot(center_probs, centers)
        dist_from_centers = [euclidean_distance(expected_center, c) for c in centers]  
        return int(np.argmin(dist_from_centers))

    return u_points.map(get_probable_center_assignment_for_one_point).collect()


def get_optimal_assignments_for_bag_indices(u_points: List[UncertainPoint], centers: List[Center],
                                            prev_assignments: Assignments, bag_indices: List[int],
                                            start_time, spark) -> Assignments:
        temp_assignments = np.array(prev_assignments)
        temp_assignments[bag_indices] = -1
        all_complete_assignments = get_all_complete_assignments(list(temp_assignments), len(centers))
        if is_time_limited(start_time):
            return all_complete_assignments[0]
        return get_best_assignments_in_list(u_points, centers, all_complete_assignments, spark)


def get_bagging_assignments(u_points: RDD, centers: List[Center], bag_size: int, spark) -> Assignments:
    start_time = time.time()
    u_points_coll = u_points.collect()
    n = len(u_points_coll)
    ep_assignments = get_expected_point_assignments(u_points, centers)

    bags_indices = [list(range(x, min(x + bag_size, n))) for x in range(0, n, bag_size)]
    opt_assignments = list(map(lambda bi: list(np.array(get_optimal_assignments_for_bag_indices(
        u_points_coll, centers, ep_assignments, bi, start_time, spark))[bi]), bags_indices))
    

    if is_time_limited(start_time):
        return None
    return reduce(lambda acc, curr: acc + curr, opt_assignments)
    

def get_bagging_assignments_with_fixed_prev_opts(u_points: RDD, centers: List[Center], bag_size: int, spark) -> Assignments:
    start_time = time.time()

    u_points_coll = u_points.collect()
    n = len(u_points_coll)
    ep_assignments = get_expected_point_assignments(u_points, centers)
    bags_indices = [list(range(x, min(x + bag_size, n))) for x in range(0, n, bag_size)]
    res = reduce(lambda acc, curr: get_optimal_assignments_for_bag_indices(u_points_coll, centers, acc, curr, start_time, spark),
                  bags_indices, ep_assignments)
    if is_time_limited(start_time):
        return None
    return res

def is_time_limited(start_time):
    return time.time() - start_time > 3600 * 4
