class BatteryModel:
    def __init__(self, capacity_kwh, max_charge_rate, max_discharge_rate, charge_efficiency, discharge_efficiency, initial_soc=0.5):
        self.capacity_kwh = capacity_kwh
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.soc = initial_soc * capacity_kwh  # Initial SOC in kWh
        self.mode = 'idle'

    def set_mode(self, mode):
        if mode not in ['idle', 'charge', 'discharge']:
            raise ValueError("Invalid mode for battery.")
        self.mode = mode

    def charge(self, available_energy):
        if self.mode == 'charge':
            charge_energy = min(available_energy, self.max_charge_rate,
                                (self.capacity_kwh - self.soc) / self.charge_efficiency)
            self.soc = min(self.soc + charge_energy * self.charge_efficiency,
                           self.capacity_kwh)  # Ensure SOC doesn't exceed capacity
            return charge_energy
        return 0

    def discharge(self, demand_left):
        if self.mode == 'discharge':
            discharge_energy = min(self.max_discharge_rate, self.soc, demand_left / self.discharge_efficiency)
            self.soc = max(self.soc - discharge_energy, 0)  # Ensure SOC doesn't fall below 0
            return discharge_energy * self.discharge_efficiency
        return 0

    def get_cost(self):
        return 0  # Assuming no cost for charging/discharging battery

    def get_state(self):
        return [self.soc, self.mode]

    def reset(self):
        self.soc = 0.5 * self.capacity_kwh
        self.mode = 'idle'
