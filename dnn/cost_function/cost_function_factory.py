from dnn.cost_function import cost_function
from dnn.cost_function.regulartization import l1_regulartization
from dnn.cost_function.regulartization import l2_regulartization

class CostFunctionFactory( ):
    @staticmethod
    def get_cost_function(c_f):
        if c_f == "l1_regulartization":
            return l1_regulartization.CostFunction
        elif c_f == "l2_regulatization":
            return l2_regulartization.CostFunction
        else:
            return cost_function.CostFunction
