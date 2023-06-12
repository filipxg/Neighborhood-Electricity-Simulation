class House:
    def __init__(self, consumption_data):
        self.consumption_data = consumption_data

    def get_consumption(self, hour):
        return self.consumption_data.iloc[hour]['Consumption']
    
    def get_prediction(self, hour):
        return self.consumption_data.iloc[hour]['Pred']