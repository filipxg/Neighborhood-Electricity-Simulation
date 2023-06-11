class Battery:
    def __init__(self, max_capacity, max_charge_rate, max_discharge_rate):
        self.current_level = 0
        self.max_capacity = max_capacity
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate

    def charge(self, amount):
        self.current_level = min(self.current_level + amount, self.max_capacity)

    def discharge(self, amount):
        amount = min(amount, self.current_level)
        self.current_level -= amount
        return amount
    