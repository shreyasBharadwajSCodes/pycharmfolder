import pandas as pd
import numpy as np
import random


class LoadModel:
    def __init__(self, load_name, current_demand, previous_demand, recovery_time):
        self.load_name = load_name
        self.current_demand = current_demand
        self.previous_demand = previous_demand
        self.recovery_time = recovery_time
        self.breakdowns = []
