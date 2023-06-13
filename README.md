# Neighborhood-Electricity-Simulation

This System Simulates the consumption of electricity and should compare it to the price to eventually reduce the grid dependency (by 70%). This, inherently means that the Demand/Supply curve reaches a 70% similarity in the curve distribution. 

TODO:
+ Implement a 24 hour basis battery system that uses the next 24 hours of the prediction consumption to define when to charge/discharge the Battery. 
+ Implement a distribution function for the change in the prediction and the true consumption.
+- As now the simulation tries to only to get the Total consumption over the day to be as closely to a straight line as possible, it is necessary to interpret the price as well. This needs to be implemented as a Price-Battery charge ratio. 

