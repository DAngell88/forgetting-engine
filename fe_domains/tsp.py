import numpy as np

class TSPDomain:
    """
    Traveling Salesman Problem (TSP) Logic.
    Validates the 200-City Benchmark.
    """
    def __init__(self, cities):
        self.cities = cities
        self.num_cities = len(cities)

    def calculate_distance(self, tour):
        """
        Calculates total Euclidean distance of the tour.
        Optimization Target: Minimize Distance.
        """
        dist = 0
        for i in range(self.num_cities):
            c1 = self.cities[tour[i]]
            c2 = self.cities[tour[(i + 1) % self.num_cities]]
            dist += np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
        return dist