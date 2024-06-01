class PriorPurchasedModel:
    def __init__(self, previous_demand, cost=3, per_of_prev=0.03):
        self.previous_demand = list(previous_demand)
        self.electricity_per_minute = [per_of_prev * x for x in self.previous_demand]
        self.cost = cost

    def get_price(self, timestep):
        #change Included timestep in parameter
        return self.cost * self.electricity_per_minute[timestep]

    def get_state(self, timestep):
        if timestep < len(self.electricity_per_minute):
            return self.electricity_per_minute[timestep]
        else:
            return 0  # Return 0 if timestep is out of bounds

    def reset(self):
        pass
