import random
import numpy as np
import pandas as pd
from typing import *
from entities import *
from config import *
from utils import *

def generate_random_coordinate(d: int) -> Coordinate:
    return list(np.random.uniform(MIN_POINT_RANGE, MAX_POINT_RANGE, d))


def generate_n_uncertain_points_in_Rd(n: int, d: int) -> List[UncertainPoint]:
    def generate_uncertain_point_inRd(d: int) -> UncertainPoint:
        z = random.randint(MIN_POINT_COUNT, MAX_POINT_COUNT)
        probs = list(list(np.random.dirichlet(np.ones(z), size=1))[0])
        return [Point(generate_random_coordinate(d), prob) for prob in probs]
        
    return [generate_uncertain_point_inRd(d) for _ in range(0, n)]


def generate_k_centers_in_Rd(k: int, d: int) -> List[Center]:
    return [generate_random_coordinate(d) for _ in range(0, k)]


def transfer_csv_to_desired_format(path: str, key: str, num_of_intervals: List[int], name='name', lat='lat', lng='lng') -> List[UncertainPoint]:
    df = pd.read_csv(path, usecols=[key, name, lat, lng])
    mins = [min(df[lat]), min(df[lng])]
    maxs = [max(df[lat]), max(df[lng])]

    centers = get_centers_of_districts(mins, maxs, num_of_intervals)
    df['district'] = df.apply(lambda x: get_index_of_nearest_center_to_coordinate([x[lat], x[lng]], centers), axis=1) 
    df1 = df.groupby([name, 'district']).agg(count=(key, 'size'), lat=(lat, 'mean'), lng=(lng, 'mean')).reset_index()
    df2 = df1.groupby([name]).agg(sum=('count', 'sum')).reset_index()
    df3 = pd.merge(df1, df2, on=name, how='inner')
    df3['district_probability'] = df3.apply(lambda x: x['count']/x['sum'], axis=1)
    df3['location_with_probability'] = df3[['district_probability', lat, lng]].values.tolist()
    df4 = df3.groupby(name)['location_with_probability'].agg(lambda x: list(x)).reset_index()

    uncertain_points = []
    for i in range(df4.shape[0]):
        points = []
        for p in df4.iloc[i][1]:
            points.append(Point(p[1:], p[0]))
        uncertain_points.append(points)
        
    return uncertain_points
  
  
def save_data_to_csv(path, dict):
    df = pd.DataFrame(dict)
    df.to_csv(path, index=False)
