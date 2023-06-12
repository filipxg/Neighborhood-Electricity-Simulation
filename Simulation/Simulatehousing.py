import os
import pandas as pd
import matplotlib.pyplot as plt
import House
import Battery
import numpy as np

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
# price_data_file = os.path.join(current_dir, 'data/Neighborhood With prices', 'price_data.csv')
hourly_consumption_file = os.path.join(current_dir, 'data', 'hourly_cons_pred_price.csv')

class System:
    def __init__(self, consumption_data_path, battery):
        self.house = House.House(pd.read_csv(consumption_data_path))
        self.battery = battery

    def average_predict(self):
        total_average_predict = 0
        for hour in range(24):
            total_average_predict += self.house.get_prediction(hour)

        total_average_predict = total_average_predict/24
        return total_average_predict

    def simulate(self):
        hourly_predict_with_battery = []
        hourly_predict_without_battery = []
        hourly_consume = []

        for day in range(len(self.house.consumption_data) // 24):
            total_average_predict = self.average_predict()

            for hour in range(24):
                # Adjusted hour index to consider multiple days
                hour_index = day * 24 + hour

                prediction = self.house.get_prediction(hour_index)

                # Without battery
                prediction_without_battery = prediction
                hourly_predict_without_battery.append(prediction_without_battery)

                # With battery
                if prediction < total_average_predict:
                    difference = total_average_predict - prediction
                    # Calculate how much we can charge the battery without exceeding its maximum capacity or the difference
                    possible_charge = min(difference, self.battery.max_charge_rate, self.battery.max_capacity - self.battery.current_level)
                    if possible_charge > 0:
                        self.battery.charge(possible_charge)
                        prediction_with_battery = prediction + possible_charge
                else:
                    difference = prediction - total_average_predict
                    # Calculate how much we can discharge from the battery without exceeding its current level or the difference
                    possible_discharge = min(difference, self.battery.max_discharge_rate, self.battery.current_level)
                    if possible_discharge > 0:
                        self.battery.discharge(possible_discharge)
                        prediction_with_battery = max(0, prediction - possible_discharge)
                    else:
                        prediction_with_battery = prediction

                hourly_predict_with_battery.append(prediction_with_battery)

        return hourly_predict_with_battery, hourly_predict_without_battery
    
    def monte_carlo_simulate(self, n_simulations):
        # Calculate the difference between predicted and actual consumption
        diff = self.house.consumption_data['Pred'] - self.house.consumption_data['Consumption']
        results = []
        for _ in range(n_simulations):
            # Generate a random sample from the 'Diff' column
            sample = diff.sample(n=len(diff), replace=True).values

            # Add this sample to the original prediction
            new_pred = self.house.consumption_data['Pred'] + sample

            # Set the new prediction to the house object
            self.house.set_prediction(new_pred)

            # Run the simulation with the new prediction
            consume_with_battery, consume_without_battery = self.simulate()
            results.append((consume_with_battery, consume_without_battery))
            
        return results


battery = Battery.Battery(max_capacity=13.5, max_charge_rate=3.3, max_discharge_rate=5)
system = System(hourly_consumption_file, battery)
results = system.monte_carlo_simulate(1000)

# Create a figure with two subplots
fig, axs = plt.subplots(2, figsize=(10, 10))

# Separate the simulations with and without battery
simulations_with_battery = [result[0] for result in results]
simulations_without_battery = [result[1] for result in results]

# Calculate the average power consumption for each hour with and without battery
avg_with_battery = np.mean(simulations_with_battery, axis=0)
avg_without_battery = np.mean(simulations_without_battery, axis=0)

# Plot each simulation and the average power consumption with battery
for simulation in simulations_with_battery:
    axs[0].plot(range(24), simulation, color='blue', alpha=0.1)  # Plot each simulation with low opacity
axs[0].plot(range(24), avg_with_battery, color='red', linewidth=2)  # Plot the average with a thicker line
axs[0].set_title('Power Consumption With Battery')
axs[0].set_xlabel('Hour')
axs[0].set_ylabel('Consumption [kWh]')

# Plot each simulation and the average power consumption without battery
for simulation in simulations_without_battery:
    axs[1].plot(range(24), simulation, color='blue', alpha=0.1)  # Plot each simulation with low opacity
axs[1].plot(range(24), avg_without_battery, color='red', linewidth=2)  # Plot the average with a thicker line
axs[1].set_title('Power Consumption Without Battery')
axs[1].set_xlabel('Hour')
axs[1].set_ylabel('Consumption [kWh]')

# Show the plots
plt.tight_layout()
plt.show()
# consume_with_battery, consume_without_battery = system.simulate()
# plt.plot(range(24), consume_without_battery, label='Without Battery')
# plt.plot(range(24), consume_with_battery, label='With Battery')
#plt.plot(range(624), real_consumption, label='Real consumption')
# plt.xlabel('Hour')
# plt.ylabel('Consumption [kWh]')
# plt.legend()
# plt.show()
