import os
import pandas as pd
import matplotlib.pyplot as plt
import House
import Battery

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
# price_data_file = os.path.join(current_dir, 'data/Neighborhood With prices', 'price_data.csv')
hourly_consumption_file = os.path.join(current_dir, 'data/Neighborhood With prices', 'house1_cons_pred_price.csv')

class System:
    def __init__(self, price_data_path, consumption_data_path, battery):
        self.price_data = pd.read_csv(price_data_path)
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

        total_average_predict = self.average_predict()

        for hour in range(24):
            prediction = self.house.get_prediction(hour)

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

        return hourly_predict_with_battery, hourly_predict_without_battery, 


battery = Battery.Battery(max_capacity=13.5, max_charge_rate=3.3, max_discharge_rate=5)
system = System(price_data_file, hourly_consumption_file, battery)
consume_with_battery, consume_without_battery = system.simulate()

plt.plot(range(24), consume_without_battery, label='Without Battery')
plt.plot(range(24), consume_with_battery, label='With Battery')
#plt.plot(range(624), real_consumption, label='Real consumption')
plt.xlabel('Hour')
plt.ylabel('Consumption [kWh]')
plt.legend()
plt.show()
