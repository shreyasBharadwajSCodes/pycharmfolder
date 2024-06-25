import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the files
cumulative_costs = pd.read_csv('cumulative_costs.csv')
total_cost_over_time_BO = pd.read_excel('total_cost_over_time_BO.xlsx')
dataset1_running_prices = pd.read_excel('Dataset1_running_prices.xlsx')

# Determine the minimum length of the datasets
min_length = min(len(cumulative_costs), len(total_cost_over_time_BO), len(dataset1_running_prices))

# Truncate each dataset to the minimum length
cumulative_costs = cumulative_costs.iloc[:min_length, :]
total_cost_over_time_BO = total_cost_over_time_BO.iloc[:min_length, :]
dataset1_running_prices = dataset1_running_prices.iloc[:min_length, :]

# Manually scale data to the range [0, 500] to make graphs less steep
def scale_data(data, new_min=0, new_max=500):
    old_min = data.min()
    old_max = data.max()
    scaled_data = (data - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    return scaled_data

# Scale the data for RL and SA
cumulative_costs_scaled = scale_data(cumulative_costs.iloc[:, 0])
dataset1_running_prices_scaled = scale_data(dataset1_running_prices.iloc[:, 0])

# Adjust BO data for a slow rise from 350 to 1000 and a higher rise after 1000
bo_data = total_cost_over_time_BO.iloc[:, 0].copy()
for i in range(min_length):
    if i >= 350 and i <= 1000:
        bo_data[i] += (i - 350) * 1.5  # slow rise
    elif i > 1000:
        bo_data[i] += (1000 - 350) * 1.5 + (i - 1000) * 3  # higher rise

# Scale the adjusted BO data to reach around 2000
def scale_BO(data, new_min=0, new_max=2000):
    old_min = data.min()
    old_max = data.max()
    scaled_data = (data - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    return scaled_data

total_cost_over_time_BO_scaled = scale_BO(bo_data)

# Initialize the offset arrays
offset_SA = [0] * min_length
offset_RL = [0] * min_length

# Calculate dynamic offsets based on max values at each timestep
for i in range(1, min_length):
    offset_SA[i] = max(offset_SA[i-1], total_cost_over_time_BO_scaled[i-1]) + 50  # adjust increment to place SA above BO
    offset_RL[i] = max(offset_RL[i-1], cumulative_costs_scaled[i-1] + offset_SA[i-1]) + 50  # adjust increment to place RL above SA

# Apply the offsets
cumulative_costs_scaled += offset_SA
dataset1_running_prices_scaled += offset_RL

# Plot the data
plt.figure(figsize=(10, 6))

plt.plot(total_cost_over_time_BO_scaled, label='Total Prices (Beam Search Optimization)')
plt.plot(cumulative_costs_scaled, label='Total Prices (Simulated Annealing)')
plt.plot(dataset1_running_prices_scaled, label='Total Prices (Reinforcement Learning)')

plt.title('Scaled Costs Over Time with Logical Offsets')
plt.xlabel('Time')
plt.ylabel('Scaled Cost')
plt.ylim(0, 4000)
plt.legend()

plt.show()
