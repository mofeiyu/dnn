import numpy as np
from basic import Cost
from cost_function import CostFunction
class L1CostFunction(Cost):
    @staticmethod
    def sum_parameters(L, parameters):
        sum_para = 0
        for l in range(1,L):
            sum_para += np.sum(np.sum(np.abs(parameters['W' + str(l)])))
        return sum_para
    @staticmethod
    def cost(AL,Y, L, parameters):
        m = AL.shape[1]
        cost = CostFunction.cost(AL,Y) + L1CostFunction.sum_parameters(L, parameters)/m
        return cost