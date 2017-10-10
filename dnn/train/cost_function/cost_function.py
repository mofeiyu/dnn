import numpy as np
from basic import Cost
class CostFunction(Cost):
    @staticmethod
    def cost(AL,Y, L = None,  parameters = None):
        m = AL.shape[1]
        cost = np.sum(np.multiply(np.log(AL), Y)) * (-1.0) / m       
        return cost