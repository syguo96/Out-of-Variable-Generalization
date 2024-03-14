from typing import Dict, Any


class DataGenSettings:

    __slots__ = (
        "num_samples",
        "split_fraction",
        "noise_var",
        "noise_skew",
        "noise_mean",
    )

    def __init__(self, num_samples, split_fraction, noise_var, noise_skew, noise_mean):
        self.num_samples = num_samples
        self.split_fraction = split_fraction
        self.noise_var = noise_var
        self.noise_skew = noise_skew
        self.noise_mean = noise_mean

    def to_dict(self) -> Dict[str, Any]:
        return {attr: getattr(self, attr) for attr in self.__slots__}

    @classmethod
    def get_default(cls):
        num_samples = 100000
        split_fraction = 0.9
        noise_var = 0.1
        noise_skew = 0
        noise_mean = 0
        return cls(num_samples, split_fraction, noise_var, noise_skew, noise_mean)
