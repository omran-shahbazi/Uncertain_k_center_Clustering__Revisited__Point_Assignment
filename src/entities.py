from typing import NewType, List
import pyspark

Coordinate = NewType('Coordinate', List[float])

class Point:
    def __init__(self, coordinate: Coordinate, prob: float) -> None:
        self.coordinate = coordinate
        self.prob = prob
    def __str__(self) -> str:
        return "Point(coordinate: {}, prob: {})".format(self.coordinate, self.prob)

UncertainPoint = NewType('UncertainPoint', List[Point])

Center = NewType('Center', Coordinate)

Assignments = NewType('Assignments', List[int])

RDD = pyspark.RDD
