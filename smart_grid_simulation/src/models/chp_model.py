class CHPModel:
    def __init__(self, min_operational_value=3, max_operational_value=20, cost_per_kwh_electricity=5):
        self.min_operational_value = min_operational_value
        self.max_operational_value = max_operational_value
        self.cost_per_kwh_electricity = cost_per_kwh_electricity
        self.current_output = 0
        self.is_off = False
        self.turn_on_time = 0
        self.current_time_step = 0

    def set_output(self, output):
        if output == 0:
            self.is_off = True
            self.turn_on_time = self.current_time_step + 90
        elif self.min_operational_value <= output <= self.max_operational_value:
            self.current_output = output
            self.is_off = False
        else:
            raise ValueError(f"CHP output must be 0 (off) or between {self.min_operational_value} and {self.max_operational_value}")

    def calculate_cost_at_current_step(self):
        self.current_time_step += 1
        if self.is_off and self.current_time_step < self.turn_on_time:
            return 0
        if self.min_operational_value <= self.current_output <= self.max_operational_value:
            return (self.current_output * self.cost_per_kwh_electricity / 60)
        return 0

    def get_state(self):
        return [self.current_output, self.is_off, self.turn_on_time]

    def reset(self):
        self.current_output = 0
        self.is_off = False
        self.turn_on_time = 0
        self.current_time_step = 0
