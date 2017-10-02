import numpy as np
from basic import Cost
class CostFunction(Cost):
    @staticmethod
    def cost(AL,Y, L, parameters):
        m = AL.shape[1]
        cost = -1 / m * np.sum(np.multiply(np.log(AL), Y))        
        return cost