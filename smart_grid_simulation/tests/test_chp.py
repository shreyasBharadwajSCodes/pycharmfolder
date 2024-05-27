import unittest
from models.chp_model import CHPModel

class TestCHPModel(unittest.TestCase):
    def setUp(self):
        self.chp = CHPModel(min_operational_value=3, max_operational_value=6, cost_per_kwh_electricity=100)

    def test_initial_conditions(self):
        self.assertEqual(self.chp.min_operational_value, 3)
        self.assertEqual(self.chp.max_operational_value, 6)
        self.assertEqual(self.chp.cost_per_kwh_electricity, 100)
        self.assertEqual(self.chp.zero_count, 0)
        self.assertFalse(self.chp.is_off)
        self.assertEqual(self.chp.turn_on_time, 0)
        self.assertEqual(self.chp.current_time_step, 0)

    def test_operational_range(self):
        cost = self.chp.calculate_cost_at_current_step(4)
        self.assertEqual(cost, 4 * 100 / 60)
        self.assertFalse(self.chp.is_off)
        self.assertEqual(self.chp.zero_count, 0)

    def test_below_operational_range(self):
        cost = self.chp.calculate_cost_at_current_step(2)
        self.assertEqual(cost, 0)
        self.assertFalse(self.chp.is_off)
        self.assertEqual(self.chp.zero_count, 0)

    def test_above_operational_range(self):
        cost = self.chp.calculate_cost_at_current_step(7)
        self.assertEqual(cost, 0)
        self.assertFalse(self.chp.is_off)
        self.assertEqual(self.chp.zero_count, 0)

    def test_turn_off(self):
        self.chp.calculate_cost_at_current_step(0)
        self.assertTrue(self.chp.is_off)
        self.assertEqual(self.chp.turn_on_time, 91)

    def test_downtime(self):
        # Turn off the CHP and check during downtime
        self.chp.turn_off()
        for i in range(89):
            cost = self.chp.calculate_cost_at_current_step(4)
            self.assertEqual(cost, 0)
            self.assertTrue(self.chp.is_off)
            self.assertEqual(self.chp.current_time_step, i + 1)

        # Check after downtime
        cost = self.chp.calculate_cost_at_current_step(4)
        self.assertEqual(cost, 4 * 100 / 60)
        self.assertFalse(self.chp.is_off)


if __name__ == '__main__':
    unittest.main()
