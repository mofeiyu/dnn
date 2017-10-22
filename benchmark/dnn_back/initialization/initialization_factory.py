from dnn.initialization.initialization_Xavier import initialization_Xavier
from dnn.initialization.initialization_ramdon import initialize_ramdon

class InitializationFactory():
    @staticmethod
    def get_initialization(i):
        if i == "initialization_Xavier":
            return initialization_Xavier
        else:
            return initialize_ramdon