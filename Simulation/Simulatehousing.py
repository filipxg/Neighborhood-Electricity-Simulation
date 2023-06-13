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
Hour24_only = os.path.join(current_dir, 'data', '24hrs_hourly_cons_pred_price.csv')

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
        cost_with_battery = 0
        cost_without_battery = 0

        total_average_predict = self.average_predict()

        for hour in range(24):
            prediction = self.house.get_prediction(hour)
            price = self.house.get_price(hour)

            # Without battery
            prediction_without_battery = prediction
            hourly_predict_without_battery.append(prediction_without_battery)
            cost_without_battery += price * prediction_without_battery 

            # Assume prediction_with_battery is same as prediction at first
            prediction_with_battery = prediction

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

            hourly_predict_with_battery.append(prediction_with_battery)
            cost_with_battery += price * prediction_with_battery

        savings = cost_without_battery - cost_with_battery
        return hourly_predict_with_battery, hourly_predict_without_battery, savings
    
    def simulate_with_price(self):
        hourly_predict_with_battery = []
        hourly_predict_without_battery = []
        cost_with_battery = 0
        cost_without_battery = 0

        prices = [self.house.get_price(hour) for hour in range(24)]
        median_price = np.median(prices)

        for hour in range(24):
            prediction = self.house.get_prediction(hour)
            price = self.house.get_price(hour)
            
            # Without battery
            prediction_without_battery = prediction
            hourly_predict_without_battery.append(prediction_without_battery)
            cost_without_battery += price * prediction_without_battery 

            # Assume prediction_with_battery is same as prediction at first
            prediction_with_battery = prediction

            # With battery
            # Charge more when price is lower
            if price < median_price:
                possible_charge = min(self.battery.max_charge_rate, self.battery.max_capacity - self.battery.current_level)
                if possible_charge > 0:
                    self.battery.charge(possible_charge)
                    prediction_with_battery = prediction + possible_charge
            # Discharge when price is higher
            else:
                possible_discharge = min(self.battery.max_discharge_rate, self.battery.current_level)
                if possible_discharge > 0:
                    self.battery.discharge(possible_discharge)
                    prediction_with_battery = max(0, prediction - possible_discharge)

            hourly_predict_with_battery.append(prediction_with_battery)
            cost_with_battery += price * prediction_with_battery

        savings = cost_without_battery - cost_with_battery
        return hourly_predict_with_battery, hourly_predict_without_battery, savings
    
    def monte_carlo_simulate(self, n_simulations, with_price):
        # Calculate the difference between predicted and actual consumption for the first 24 hours
        diff = self.house.consumption_data['Pred'].iloc[:24] - self.house.consumption_data['Consumption'].iloc[:24]
        results = []
        total_savings = 0
        for _ in range(n_simulations):
            # Generate a random sample from the 'Diff' column for the first 24 hours
            sample = diff.sample(n=24, replace=True).values

            # Add this sample to the original prediction for the first 24 hours
            new_pred = self.house.consumption_data['Pred'].iloc[:24] + sample

            # Set the new prediction to the house object for the first 24 hours
            self.house.set_prediction(new_pred)

            # Run the simulation with the new prediction
            if with_price == True:
                consume_with_battery, consume_without_battery, savings = self.simulate_with_price()
            elif with_price == False:
                consume_with_battery, consume_without_battery, savings = self.simulate()
            
            results.append((consume_with_battery, consume_without_battery))
            total_savings += savings

        return results, total_savings


battery = Battery.Battery(max_capacity=13.5, max_charge_rate=3.3, max_discharge_rate=5)
system = System(hourly_consumption_file, battery)

results_linear, total_savings_linear = system.monte_carlo_simulate(1000, False)
print(total_savings_linear)

results_price, total_savings_price = system.monte_carlo_simulate(1000, True)
print(total_savings_price)

# Separate the simulations with and without battery
simulations_with_battery_linear = [result[0] for result in results_linear]
simulations_without_battery_linear = [result[1] for result in results_linear]

simulations_with_battery_price = [result[0] for result in results_price]
simulations_without_battery_price = [result[1] for result in results_price]
# print(len(simulations_without_battery))
# print(simulations_with_battery)

# Calculate the average power consumption for each hour with and without battery
avg_with_battery_linear = np.mean(simulations_with_battery_linear, axis=0)
avg_without_battery_linear = np.mean(simulations_without_battery_linear, axis=0)

avg_with_battery_price = np.mean(simulations_with_battery_price, axis=0)
avg_without_battery_price = np.mean(simulations_without_battery_price, axis=0)

# Create a figure with 4 subplots
fig, axs = plt.subplots(2,2, figsize=(20, 10))

# Plot each simulation and the average power consumption for the original strategy
for simulation in simulations_with_battery_linear:
    axs[0,0].plot(range(len(simulation)), simulation, color='blue', alpha=0.07)  # Plot each simulation with low opacity
axs[0,0].plot(range(len(avg_with_battery_linear)), avg_with_battery_linear, color='red', linewidth=2)   # Plot the average with a thicker line
axs[0,0].set_title('Power Consumption With Battery (Average-Based Strategy)')
axs[0,0].set_xlabel('Hour')
axs[0,0].set_ylabel('Consumption [kWh]')

for simulation in simulations_without_battery_linear:
    axs[0,1].plot(range(len(simulation)), simulation, color='blue', alpha=0.07)  # Plot each simulation with low opacity
axs[0,1].plot(range(len(avg_without_battery_linear)), avg_without_battery_linear, color='red', linewidth=2)  # Plot the average with a thicker line
axs[0,1].set_title('Power Consumption Without Battery (Average-Based Strategy)')
axs[0,1].set_xlabel('Hour')
axs[0,1].set_ylabel('Consumption [kWh]')

# Plot each simulation and the average power consumption for the price strategy
for simulation in simulations_with_battery_price:
    axs[1,0].plot(range(len(simulation)), simulation, color='blue', alpha=0.07)  # Plot each simulation with low opacity
axs[1,0].plot(range(len(avg_with_battery_price)), avg_with_battery_price, color='red', linewidth=2)  # Plot the average with a thicker line
axs[1,0].set_title('Power Consumption Without Battery (Price-Based Strategy)')
axs[1,0].set_xlabel('Hour')
axs[1,0].set_ylabel('Consumption [kWh]')

for simulation in simulations_without_battery_price:
    axs[1,1].plot(range(len(simulation)), simulation, color='blue', alpha=0.07)  # Plot each simulation with low opacity
axs[1,1].plot(range(len(avg_without_battery_price)), avg_without_battery_price, color='red', linewidth=2)  # Plot the average with a thicker line
axs[1,1].set_title('Power Consumption Without Battery (Price-Based Strategy)')
axs[1,1].set_xlabel('Hour')
axs[1,1].set_ylabel('Consumption [kWh]')

# data = pd.read_csv(hourly_consumption_file)

# Plot Consumption, Pred, and Price
# axs[2].plot(data.index[:24], data['Consumption'][:24], label='Consumption', color='blue')
# axs[2].plot(data.index[:24], data['Pred'][:24], label='Pred', color='green')
# axs[2].set_title('Consumption vs. Prediction')
# axs[2].set_xlabel('Time')
# axs[2].set_ylabel('kWh')
# axs[2].legend()

# axs[3].plot(data.index[:24], data['Price'][:24], label='Price', color='red')
# axs[3].set_title('Price of electricity')
# axs[3].set_xlabel('Time')
# axs[3].set_ylabel('kWh')
# axs[3].legend()

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
