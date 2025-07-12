from abc import ABC

class ObservationState(ABC):
    """base object class for the observation state"""
    def __init__(self) -> None:
        super().__init__()
    
    def _normalize(self):
        '''all the observation states should be normalized
        some will be normalized with min-max scaling, while some will be z-normalized
        -> we know most of the ranges based on prior knowledge, so we can compute the expected
        mean and standard deviation given the range, drawn from a uniform distribution 
        **********************************************************************************
        E[X] = xf(x)의 a에서 b까지의 적분 계산 결과
        V[X]=E[X^2]-(E[X])^2
        uniform distribution에서 f(x) = 1/(b-a) if a <= x <= b else 0
        E[X] = (a+b)/2 
        V[X] = ((b-a)^2)/12
        '''
        pass
    
    def _reset(self):
        pass
    
    


