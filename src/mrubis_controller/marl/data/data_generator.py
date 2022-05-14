import random
import numpy

shop_name = "mRUBiS #"
component_names = ["Availability Item Filter",
                   "Bid and Buy Service",
                   "Seller Reputation Item Filter",
                   "User Management Service",
                   "Reputation Service",
                   "Comment Item Filter",
                   "Recommendation Item Filter",
                   "Authentication Service",
                   "Persistence Service",
                   "Item Management Service",
                   "Buy Now Item Filter",
                   "Query Service",
                   "Past Sales Item Filter",
                   "Inventory Service",
                   "Region Item Filter",
                   "Category Item Filter",
                   "Last Second Sales Item Filter",
                   "Future Sales Item Filter"]


class DataGenerator:
    def __init__(self, seed_shop_dependencies=1, number_of_shops=3):
        self.number_of_shops = number_of_shops
        # [[number_of_failures, offset, use_dependency]]
        self.shop_config = []
        self.shop_dep = self._generate_failing_dependencies(seed_shop_dependencies)

    def _generate_failing_dependencies(self, seed):
        """ Returns for each component index a sorted list of most probable component indices
            which would fail in the observations.
            This should simulate the shop internal component dependencies.
        """
        rng = numpy.random.RandomState(seed)
        dependencies = []
        for _ in range(18):
            probabilities = {i: rng.uniform(0, 1) for i in range(18)}
            dependencies.append(sorted(probabilities, key=probabilities.get, reverse=True))
        return dependencies

    def _generate_shop(self, failing_component, failure_name="CF2"):
        """ returns a shop with only currently necessary attributes per component """
        shop = {}
        shop_utility = 0
        for component in component_names:
            shop[component] = {
                "failure_name": failure_name if component in failing_component else "None",
                "component_utility": random.uniform(3000, 25000),
                "criticality": random.uniform(10, 30),
                "connectivity": random.uniform(1, 10),
                "reliability": 0.5,
                "importance": 12.0,
                "provided_interface": 1,
                "required_interface": random.randint(0, 5),
                "adt": random.uniform(1, 1.5),
                "perf_max": random.uniform(5, 20),
                "sat_point": random.uniform(0, 0.2),
                "replica": random.uniform(1, 30),
                "request": random.uniform(50, 450),
            }
            shop_utility += shop[component]["component_utility"]
        for component in component_names:
            shop[component]["shop_utility"] = shop_utility

        return shop

    def _generate_state_fails(self):
        """ returns actual failing component indices """
        return [random.randint(0, 17) for _ in range(self.number_of_shops)]

    def _get_offset_fails(self, failing_component_index, number_of_failures, offset):
        """ returns a list of false failing components starting with an offset of the actual failing component """
        return [component_names[(failing_component_index + offset + j) % 18] for j in
                range(number_of_failures)]

    def _get_dependency_fails(self, failing_component_dependencies, number_of_failures, offset):
        """ returns most probable false failing components
            the offset adds a randomness factor
        """
        failing_component_indices = random.sample(failing_component_dependencies[:(number_of_failures + offset)],
                                                  number_of_failures)
        return [component_names[index] for index in failing_component_indices]

    def set_shop_config(self, config=None):
        """ different configs:  [[number_of_failures, offset, use_dependency]]
            one config:         [number_of_failures, offset, use_dependency]
        """

        if len(numpy.shape(config)) > 1:
            self.shop_config = config
        else:
            if config:
                self.shop_config = [config for i in range(0, self.number_of_shops)]
            else:
                self.shop_config = [[random.randint(1, self.number_of_shops - 1),
                                     random.randint(1, self.number_of_shops - 1),
                                     bool(random.getrandbits(1))] for i in range(0, self.number_of_shops)]

    def generate_shops_with_failures(self, seed=1):
        """ returns shop observations and the actual failing component per shop """
        # TODO return ranking component failures
        shops_obs = {}
        shops_state = {}
        failing_component_indices = self._generate_state_fails()
        for i in range(1, 1 + self.number_of_shops):
            if self.shop_config[i - 1][2]:
                obs_fails = self._get_dependency_fails(self.shop_dep[failing_component_indices[i - 1]],
                                                       self.shop_config[i - 1][0], self.shop_config[i - 1][1])
            else:
                obs_fails = self._get_offset_fails(failing_component_indices[i - 1],
                                                   self.shop_config[i - 1][0], self.shop_config[i - 1][1])
            shops_obs[shop_name + str(i)] = self._generate_shop(obs_fails)
            shops_state[shop_name + str(i)] = component_names[failing_component_indices[i - 1]]

        return shops_obs, shops_state


if __name__ == "__main__":
    generator = DataGenerator(seed_shop_dependencies=1)
    generator.set_shop_config([2, 3, True])
    generator.generate_shops_with_failures()
