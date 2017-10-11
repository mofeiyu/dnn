from dnn.initialization.initialization_he import initialization_he
from dnn.initialization.initialization_ramdon import initialize_ramdon

class InitializationFactory():
    @staticmethod
    def get_initialization(i):
        if i == "initialization_he":
            return initialization_he
        else:
            return initialize_ramdon