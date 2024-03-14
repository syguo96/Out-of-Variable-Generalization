class DataGenerationSettings:

    def __init__(self, num_samples, split_fraction, noise_var, noise_skew, noise_mean):
        self.num_samples = num_samples
        self.split_fraction = split_fraction
        self.noise_var = noise_var
        self.noise_skew = noise_skew
        self.noise_mean = noise_mean

    @classmethod
    def get_default(cls):
        num_samples = 100000
        split_fraction = 0.9
        noise_var = 0.1
        noise_skew = 0
        noise_mean = 0
        return cls(num_samples, split_fraction, noise_var, noise_skew, noise_mean)
