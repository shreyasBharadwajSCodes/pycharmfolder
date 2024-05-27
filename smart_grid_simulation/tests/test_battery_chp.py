import unittest
from smart_grid_simulation.src.models.chp_model import CHPModel
from smart_grid_simulation.src.models.battery_model import BatteryModel
class TestModels(unittest.TestCase):
    def setUp(self):
        # Initialize instances of the models with sample parameters
        self.chp_model = CHPModel(min_operational_value=10, max_operational_value=50, cost_per_kwh_electricity=0.1, cost_penalty_zero=100)
        self.battery_model = BatteryModel(capacity_kwh=50, charge_efficiency=0.9, discharge_efficiency=0.9,
                                          max_charge_rate=10, max_discharge_rate=10, initial_soc=0.5,
                                          degradation_per_cycle=0.05)

    def test_chp_model(self):
        # Test valid electricity usage within operational range
        self.assertEqual(self.chp_model.calculate_cost_at_current_step(20), 2)  # 20 kWh * $0.1 / 60 = 0.03333 per min

        # Test invalid electricity usage outside operational range
        with self.assertRaises(ValueError):
            self.chp_model.calculate_cost_at_current_step(5)  # Below min_operational_value

        # Test penalty for consecutive zero occurrences
        for _ in range(90):
            self.assertEqual(self.chp_model.calculate_cost_at_current_step(0), 100)  # Penalty for 0 usage
        with self.assertRaises(ValueError):
            self.chp_model.calculate_cost_at_current_step(0)  # Exceeded consecutive zero limit

    def test_battery_model(self):
        # Test valid charging and discharging operations
        self.assertEqual(self.battery_model.battery_step('charging'), 25)  # Initial SoC 50% * 50 kWh = 25 kWh
        self.assertEqual(self.battery_model.battery_step('discharging'), 20)  # Discharge 5 kWh

        # Test exceeding maximum charge rate
        self.assertEqual(self.battery_model.battery_step('charging'), 25)  # Charge 5 kWh at max rate

        # Test exceeding maximum discharge rate
        self.assertEqual(self.battery_model.battery_step('discharging'), 15)  # Discharge 10 kWh at max rate

    def tearDown(self):
        # Clean up resources after running the test cases
        del self.chp_model
        del self.battery_model

if __name__ == '__main__':
    unittest.main()
