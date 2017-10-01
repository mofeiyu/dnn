import numpy as np
from init_ import Cost
class L2_Cost_Function(Cost):
    @staticmethod
    def l2_cost(AL,Y, parameters):
        
        return cost