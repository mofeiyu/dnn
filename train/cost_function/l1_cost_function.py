import numpy as np
from basic import Cost
from cost_funtion import CostFunction
class L1CostFunction(Cost):
    @staticmethod
    def sum_parameters(L,parameters):
        sum_para = 0
        for l in range(1,L):
            sum_para += np.sum(np.sum(np.abs(parameters['W' + str(l)])))
        return sum_para
    @staticmethod
    def cost(AL,Y, L, parameters):
        cost = cost_funtion.CostFunction(AL,Y) + sum_parameters(L,parameters)
        return cost