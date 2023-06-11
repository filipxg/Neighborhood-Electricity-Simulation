import pandas as pd
import matplotlib.pyplot as plt
import House
import Battery

class System:
    def __init__(self, price_data_path, consumption_data_path, battery):
        self.price_data = pd.read_csv(price_data_path)
        self.house = House.House(pd.read_csv(consumption_data_path))
        self.battery = battery

    def average_consume(self):
        total_average_consume = 0
        for hour in range(24):
            total_average_consume += self.house.get_consumption(hour)

        total_average_consume = total_average_consume/24
        return total_average_consume

    def simulate(self):
        hourly_consume_with_battery = []
        hourly_consume_without_battery = []

        total_average_consume = self.average_consume()

        for hour in range(24):
            consumption = self.house.get_consumption(hour)

            # Without battery
            consumption_without_battery = consumption
            hourly_consume_without_battery.append(consumption_without_battery)

            # With battery
            if consumption < total_average_consume:
                difference = total_average_consume - consumption
                # Calculate how much we can charge the battery without exceeding its maximum capacity or the difference
                possible_charge = min(difference, self.battery.max_charge_rate, self.battery.max_capacity - self.battery.current_level)
                if possible_charge > 0:
                    self.battery.charge(possible_charge)
                    consumption_with_battery = consumption + possible_charge
            else:
                difference = consumption - total_average_consume
                # Calculate how much we can discharge from the battery without exceeding its current level or the difference
                possible_discharge = min(difference, self.battery.max_discharge_rate, self.battery.current_level)
                if possible_discharge > 0:
                    self.battery.discharge(possible_discharge)
                    consumption_with_battery = max(0, consumption - possible_discharge)
                else:
                    consumption_with_battery = consumption

            hourly_consume_with_battery.append(consumption_with_battery)

        return hourly_consume_with_battery, hourly_consume_without_battery, 



battery = Battery.Battery(max_capacity=10000, max_charge_rate=2, max_discharge_rate=2)
system = System('data/price_data.csv', 'data/average_hourly_consumption.csv', battery)
consume_with_battery, consume_without_battery = system.simulate()

plt.plot(range(24), consume_without_battery, label='Without Battery')
plt.plot(range(24), consume_with_battery, label='With Battery')
plt.xlabel('Hour')
plt.ylabel('Consumption [kWh]')
plt.legend()
plt.show()
