import pandas as pd
import numpy as np


class Mechanism:
    def __init__(self, epsilon=1 / 2):
        if epsilon <= 0:
            raise Exception("Epsilon cannot be 0 or negative.")
        self.budget = 0
        self.epsilon = epsilon

    def noise(self, shape=None, sensitivity=1):
        raise Exception(
            "Cannot noise from base class! Please specify a mechanism= such as mechanism=LaplaceMechanism()."
        )


class GaussianMechanism(Mechanism):
    def __init__(self, epsilon=1 / 2, delta=None):
        super().__init__(epsilon)
        self.delta = delta

    def noise(self, shape=None, sensitivity=1):
        return np.random.normal(0, self.normal_variance(sensitivity, shape), size=shape)

    def normal_variance(self, sensitivity, shape):
        """
        N (0, 2 ln(1.25/δ)∆_2^2/ε^2)
        """
        if self.delta:
            delta = self.delta
        else:
            delta = 1/shape[0]
        return (2 * np.log(1.25 / delta) * (sensitivity**2)) / (
            self.epsilon**2
        )
    # mean: sensitivity is sqrt(d)/n


class LaplaceMechanism(Mechanism):
    def __init__(self, epsilon=1 / 2):
        super().__init__(epsilon)

    def noise(self, shape=None, sensitivity=1):
        return np.random.laplace(0, self.laplace_scale(sensitivity), size=shape)

    def laplace_scale(self, sensitivity):
        """
        ∆ / ε
        """
        return sensitivity / self.epsilon


@pd.api.extensions.register_series_accessor("private")
class PrivateSeriesAccessor:
    def __init__(self, pandas_obj: pd.Series):
        self.series: pd.Series = pandas_obj

    def noise(self, mechanism=Mechanism()) -> pd.Series:
        return self.series + mechanism.noise(shape=self.series.shape)

    def mean(self, mechanism=Mechanism()) -> float:
        sensitivity = 1 / max(1, len(self.series)) # safe div zero
        series_noised = self.series + mechanism.noise(
            shape=self.series.shape, sensitivity=sensitivity
        )
        return series_noised.mean()

    def count(self, mechanism=Mechanism()) -> float:
        pass

    def sum(self, mechanism=Mechanism()) -> float:
        pass

    def variance(self, mechanism=Mechanism()) -> float:
        pass

    def std(self, mechanism=Mechanism()) -> float:
        pass

    def quantile(self, mechanism=Mechanism()) -> float:
        pass

    def hist(self, mechanism=Mechanism(), bins=None) -> tuple[list[float], list[float]]:
        hist, bins = np.histogram(self.series, bins=bins)
        hist_noised = hist + mechanism.noise(shape=hist.shape)
        return hist_noised, bins


@pd.api.extensions.register_dataframe_accessor("private")
class PrivateDfAccessor:
    def __init__(self, pandas_obj: pd.DataFrame):
        self.series: pd.core.series.Series = pandas_obj

    def noise(self, mechanism=Mechanism()) -> pd.DataFrame:
        return self.series + mechanism.noise(shape=self.series.shape)
