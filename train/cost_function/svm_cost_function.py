import numpy as np
from init_ import Cost
class svm_Cost_Function(Cost):
    @staticmethod
    def svm_cost(AL,Y, parameters):
        
        return cost