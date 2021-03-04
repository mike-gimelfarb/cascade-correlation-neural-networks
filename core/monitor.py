import math
from collections import OrderedDict


class StoppingMonitor:
    
    def initialize(self):
        raise NotImplementedError
    
    def update(self, loss):
        raise NotImplementedError


class EarlyStoppingMonitor(StoppingMonitor):
    
    def __init__(self, min_delta, patience, max_iters, normalize=False):
        self.min_delta = max(min_delta, 0.0)
        self.patience = max(patience, 1)
        self.max_iters = max(max_iters, 1)
        self.normalize = normalize
        
    def initialize(self):
        self.t = 0
        self.best_loss = math.inf
        self.epochs_without_improvement = 0
        
    def update(self, loss):
        self.t += 1
        improvement = self.best_loss - loss
        if self.normalize:
            improvement /= max(abs(loss), 1e-15)
        self.best_loss = min(loss, self.best_loss)
        if improvement < self.min_delta:
            self.epochs_without_improvement += 1
        else:
            self.epochs_without_improvement = 0
        return self.t >= self.max_iters or self.epochs_without_improvement > self.patience

class TargetStoppingMonitor(StoppingMonitor):
    
    def __init__(self, target_loss, max_iters):
        self.target_loss = target_loss
        self.max_iters = max(max_iters, 1)
    
    def initialize(self):
        self.t = 0
        self.best_loss = math.inf
    
    def update(self, loss):
        self.t += 1
        self.best_loss = min(loss, self.best_loss)
        return self.t >= self.max_iters or self.best_loss <= self.target_loss
    
        
class LossHistoryMonitor:
    
    def __init__(self, keywords=['loss', 'valid_loss', 'metric', 'valid_metric']):
        self.keywords = keywords
        self.clear()
        
    def append(self, **kwargs):
        for key, value in kwargs.items():
            self.data[key].append(value)
    
    def summary(self, count=1, format_str=' {:.8f}'):
        result = ''
        for i in range(1, count + 1):
            result += ', '.join(map(str, [k + format_str.format(v[-i]) for k, v in self.data.items()])) + '\n'
        return result
        
    def clear(self):
        self.data = OrderedDict(zip(self.keywords, [[] for _ in self.keywords]))
        
