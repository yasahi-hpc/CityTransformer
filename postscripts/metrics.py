import abc
import numpy as np

class _BaseMetric(abc.ABC):
    """
    Abstract metric class
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {
                          'factor',
                          'levelset',
                          'epsilon',
                          'threshold',
                         }

        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood: ', kwarg)

        self.factor = kwargs.get('factor', 2)
        self.epsilon = kwargs.get('epsilon', 1.e-4)
        self.levelset = kwargs.get('levelset')
        self.threshold = kwargs.get('threshold', 0.9999999e-8)

        if self.levelset is None:
            raise ValueError('Argument levelset must be given')

        self.object_mask = self.levelset < 0

    @abc.abstractmethod
    def evaluate(self, pred, obs):
        raise NotImplementedError()

class Fraction(_BaseMetric):
    """
    Fraction metric: FAC2, FAC5 etc
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def evaluate(self, pred, obs):
        target_area = np.logical_and(obs > 0., self.object_mask )

        fraction = np.where(target_area, pred/obs, 0)

        count = np.logical_and( fraction >= 1/self.factor, fraction <= self.factor )

        factor = np.sum(count) / np.sum(target_area)

        return factor
    
class FB(_BaseMetric):
    """
    Best: FB == 0
    Bad: FB > 0 under estimateion, FB < 0 over estimation
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def evaluate(self, pred, obs):
        pred = np.where(self.object_mask, pred, 0)
        obs  = np.where(self.object_mask, obs, 0)

        count = np.sum(self.object_mask)
        mean = lambda var: np.sum(var) / count
         
        return 2 * (mean(obs) - mean(pred)) / (mean(obs) + mean(pred))

class NAD(_BaseMetric):
    """
    threshold-based normalized absolute difference (NAD)
    NAD = A_F / (A_F + A_OV)

    A_F: average number of false negative and false positive pairs
    A_OV: number of valid pairs (C_p > C_threshold and C_r > C_threshold)
    Best: NAD == 0
    Bad: NAD >> 0
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def evaluate(self, pred, obs):
        # Firstly, exclude inside objects
        pred = pred[self.object_mask]
        obs = obs[self.object_mask]
        
        # negative values are considered as zeros
        valid = np.logical_and(pred > self.threshold, obs > self.threshold)
        false_positive = np.logical_and(pred > self.threshold, obs < self.threshold)
        false_negative = np.logical_and(pred < self.threshold, obs > self.threshold)
        zero_zero = np.logical_and(pred <= self.threshold, obs <= self.threshold) # Unused

        A_F = ( np.sum(false_positive) + np.sum(false_negative) ) / 2
        A_OV = np.sum(valid)

        return A_F / (A_F + A_OV)

class MG(_BaseMetric):
    """
    Geometric Mean
    MG = exp( mean(ln(Cr)) - mean(ln(Cp)) )
    Best: MG == 1
    Bad: MG << 1
    [RK] The average operations are performed over the grid points where 
    the both reference and predicted concenration are larger than 0
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(self, pred, obs):
        # Firstly, exclude inside objects
        pred = np.where(self.object_mask, pred, 0)
        obs  = np.where(self.object_mask, obs, 0)

        target_area = np.logical_and(pred > 0, obs > 0)
        pred = np.log(pred, where=target_area, out=np.zeros_like(pred))
        obs  = np.log(obs,  where=target_area, out=np.zeros_like(obs))

        count = np.sum(target_area)
        mean = lambda var: np.sum(var) / count

        return np.exp(mean(pred) - mean(obs))

class VG(_BaseMetric):
    """
    Geometric Variance
    VG = exp( mean( (ln(Cr)-ln(Cp))^2 ) )

    Best: VG == 1
    Bad: VG >> 1
    [RK] The average operations are performed over the grid points where 
    the both reference and predicted concenration are larger than 0
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(self, pred, obs):
        # Firstly, exclude inside objects
        pred = np.where(self.object_mask, pred, 0)
        obs  = np.where(self.object_mask, obs, 0)

        target_area = np.logical_and(pred > 0, obs > 0)
        pred = np.log(pred, where=target_area, out=np.zeros_like(pred))
        obs  = np.log(obs,  where=target_area, out=np.zeros_like(obs))
        
        count = np.sum(target_area)
        mean = lambda var: np.sum(var) / count
        
        return np.exp( mean( (pred - obs)**2 ) )

class PCC(_BaseMetric):
    """
    Pearson's correlation coefficient (PCC)
    Best: PCC == 1
    Bad: PCC << 1
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def evaluate(self, pred, obs):
        # Firstly, exclude inside objects
        pred = np.where(self.object_mask, pred, 0)
        obs  = np.where(self.object_mask, obs, 0)
    
        numerator = np.sum( (pred - np.mean(pred) ) * (obs - np.mean(obs) ) )
        denominator = np.sqrt( np.sum( (pred - np.mean(pred))**2 ) ) * np.sqrt( np.sum( (obs - np.mean(obs))**2 ) )
    
        return numerator / denominator

def get_metric(name):
    METRICS = {
        'FAC2': Fraction,
        'FAC5': Fraction,
        'MG': MG,
        'VG': VG,
        'PCC': PCC,
        'NAD': NAD,
        'FB': FB,
    }

    for n, m in METRICS.items():
        if n.lower() == name.lower():
            return m
    
    raise ValueError(f'metric {name} is not defined')
