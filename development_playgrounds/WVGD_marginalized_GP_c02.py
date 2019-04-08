import numpy as np
import matplotlib.pyplot as plt

from brancher.variables import ProbabilisticModel, Ensemble

from brancher.stochastic_processes import GaussianProcess as GP
from brancher.stochastic_processes import SquaredExponentialCovariance as SquaredExponential
from brancher.stochastic_processes import WhiteNoiseCovariance as WhiteNoise
from brancher.stochastic_processes import HarmonicCovariance as Harmonic
from brancher.stochastic_processes import ConstantMean
from brancher.variables import RootVariable
from brancher.standard_variables import NormalVariable as Normal
from brancher.standard_variables import LogNormalVariable as LogNormal
from brancher.inference import WassersteinVariationalGradientDescent as WVGD
from brancher import inference
from brancher.visualizations import plot_particles, plot_density

from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from scipy.stats import zscore


def load_mauna_loa_atmospheric_co2():
    ml_data = fetch_openml(data_id=41187)
    months = []
    ppmv_sums = []
    counts = []

    y = ml_data.data[:, 0]
    m = ml_data.data[:, 1]
    month_float = y + (m - 1) / 12
    ppmvs = ml_data.target

    for month, ppmv in zip(month_float, ppmvs):
        if not months or month != months[-1]:
            months.append(month)
            ppmv_sums.append(ppmv)
            counts.append(1)
        else:
            # aggregate monthly sum to produce average
            ppmv_sums[-1] += ppmv
            counts[-1] += 1

    months = np.asarray(months).reshape(-1, 1)
    avg_ppmvs = np.asarray(ppmv_sums) / counts
    return months, avg_ppmvs


X, y = load_mauna_loa_atmospheric_co2()

num_datapoints = 25
x_range = X.flatten()[:num_datapoints]
data = np.reshape(zscore(y[:num_datapoints]), newshape=(1, num_datapoints))
x = RootVariable(x_range, name="x")

# Model
length_scale = LogNormal(0., 0.3, name="length_scale")
noise_var = LogNormal(0., 0.3, name="noise_var")
amplitude = LogNormal(0., 0.3, name="amplitude")
freq = LogNormal(loc=0.5, scale=0.5, name="freq")
mu = ConstantMean(0.5)
cov = SquaredExponential(scale=length_scale)*Harmonic(frequency=freq)*amplitude + WhiteNoise(magnitude=noise_var)
f = GP(mu, cov, name="f")
y = f(x)
model = ProbabilisticModel([y])

# Observe data
y.observe(data)
plt.plot(x_range.flatten(), data.flatten())
plt.show()


# Variational model
num_particles = 3
initial_locations = [(np.random.normal(0.8, 0.2), np.random.normal(0.8, 0.2),
                      np.random.normal(0.8, 0.2), np.random.normal(0.8, 0.2))
                     for _ in range(num_particles)]
particles = [ProbabilisticModel([RootVariable(location[0], name="length_scale", learnable=True),
                                 RootVariable(location[1], name="noise_var", learnable=True),
                                 RootVariable(location[2], name="amplitude", learnable=True),
                                 RootVariable(location[2], name="freq", learnable=True)])
             for location in initial_locations]

# Importance sampling distributions
variational_samplers = [ProbabilisticModel([LogNormal(np.log(location[0]), 0.3, name="length_scale", learnable=True),
                                            LogNormal(np.log(location[1]), 0.3, name="noise_var", learnable=True),
                                            LogNormal(np.log(location[2]), 0.3, name="amplitude", learnable=True),
                                            LogNormal(np.log(location[3]), 0.3, name="freq", learnable=True)])
                        for location in initial_locations]

# Inference
inference_method = WVGD(variational_samplers=variational_samplers,
                        particles=particles,
                        biased=False)
inference.perform_inference(model,
                            inference_method=inference_method,
                            number_iterations=600,
                            number_samples=50,
                            optimizer="SGD",
                            lr=0.0025,
                            posterior_model=particles,
                            pretraining_iterations=0)
loss_list = model.diagnostics["loss curve"]
plt.plot(loss_list)
plt.show()

print(inference_method.weights)

final_ensemble = Ensemble(variational_samplers, inference_method.weights)
# Posterior plot
plot_density(final_ensemble, ["length_scale", "noise_var", "amplitude", "freq"], number_samples=3000)
plt.show()