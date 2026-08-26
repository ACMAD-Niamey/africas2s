from abc import ABC, abstractmethod

class MetricBase(ABC):
    @abstractmethod
    def compute(self, forecast, obs, **kwargs):
        ...
