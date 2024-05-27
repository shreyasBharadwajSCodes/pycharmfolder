class PublicGridModel:
    def __init__(self, pg_tolerance=0.02, basic_cost=200, big_cost_penalty=1000):
        self.pg_tolerance = pg_tolerance
        self.basic_cost = basic_cost
        self.big_cost_penalty = big_cost_penalty
        self.usage = 0

    def get_price(self, demand_left, total_demand):
        if demand_left <= 0:
            return 0
        ratio = demand_left / total_demand
        self.usage = demand_left  # Ensure this correctly updates the usage
        if ratio <= self.pg_tolerance:
            return self.basic_cost * ratio * 100
        else:
            return ratio * self.big_cost_penalty * 100

    def get_state(self):
        return [self.usage, self.pg_tolerance]

    def reset(self):
        self.usage = 0

'''
class PublicGrid:
    def __init__(self, penalty_rate):
        self.penalty_rate = penalty_rate

    def get_price(self, demand_left, total_demand):
        if demand_left <= 0:
            return 0
        penalty = demand_left / total_demand * self.penalty_rate
        return penalty

    def get_state(self):
        return [0]  # Placeholder for state

'''