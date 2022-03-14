import numpy as np

class _BaseMetric:
    """
    Abstract metric class
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {
                          'factor',
                          'levelset',
                          'epsilon',
                         }

        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood: ', kwarg)

        self.factor = kwargs.get('factor', 2)
        self.epsilon = kwargs.get('epsilon', 1.e-4)
        self.levelset = kwargs.get('levelset')

        if self.levelset is None:
            raise ValueError('Argument levelset must be given')

        self.object_mask = self.levelset < 0

    def _evaluate(self, pred, obs):
        raise NotImplementedError()

    def evaluate(self, pred, obs):
        return self._evaluate(pred, obs)


class Fraction(_BaseMetric):
    """
    Fraction metric: FAC2, FAC5 etc
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def _evaluate(self, pred, obs):
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
        
    def _evaluate(self, pred, obs):
        pred = np.where(self.object_mask, pred, 0)
        obs  = np.where(self.object_mask, obs, 0)

        return 2 * (np.mean(obs) - np.mean(pred)) / (np.mean(obs) + np.mean(pred))

class NAD(_BaseMetric):
    """
    threshold-based normalized absolute difference (NAD)
    Best: NAD == 0
    Bad: NAD >> 0
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def _evaluate(self, pred, obs):
        # Firstly, exclude inside objects
        pred = pred[self.object_mask]
        obs = obs[self.object_mask]
        
        # negative values are considered as zeros
        self.valid = np.logical_and(pred > 0, obs > 0)
        self.false_positive = np.logical_and(pred >  0, obs <= 0)
        self.false_negative = np.logical_and(pred <= 0, obs > 0)
        self.zero_zero = np.logical_and(pred <= 0, obs <= 0)
        
        A_F = np.sum(self.false_positive) + np.sum(self.false_negative)
        A_OV = np.sum(self.valid) + np.sum(self.zero_zero)
        
        return A_F / (A_F + A_OV)

class MG(_BaseMetric):
    """
    Geometric Mean
    Best: MG == 1
    Bad: MG << 1
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _evaluate(self, pred, obs):
        # Firstly, exclude inside objects
        pred = np.where(self.object_mask, pred, 0)
        obs  = np.where(self.object_mask, obs, 0)

        pred = np.log(pred, where=pred>0, out=np.zeros_like(pred))
        obs  = np.log(obs,  where=obs>0,  out=np.zeros_like(obs))

        return np.exp(np.mean(pred) - np.mean(obs))

class VG(_BaseMetric):
    """
    Geometric Variance
    Best: VG == 1
    Bad: VG >> 1
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _evaluate(self, pred, obs):
        # Firstly, exclude inside objects
        pred = np.where(self.object_mask, pred, 0)
        obs  = np.where(self.object_mask, obs, 0)

        pred = np.log(pred, where=pred>0, out=np.zeros_like(pred))
        obs  = np.log(obs,  where=obs>0,  out=np.zeros_like(obs))

        return np.exp(np.mean(pred - obs)**2)

def get_metric(name):
    METRICS = {
        'FAC2': Fraction,
        'FAC5': Fraction,
        'MG': MG,
        'VG': VG,
        'NAD': NAD,
        'FB': FB,
    }

    for n, m in METRICS.items():
        if n.lower() == name.lower():
            return m
    
    raise ValueError(f'metric {name} is not defined')
