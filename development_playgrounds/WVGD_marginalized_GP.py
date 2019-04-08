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
import brancher.functions as BF

num_datapoints = 23 #25
x_range = np.linspace(0, 2., num_datapoints)
x = RootVariable(x_range, name="x")

# Model
length_scale = LogNormal(0., 0.3, name="length_scale")
noise_var = LogNormal(0., 0.3, name="noise_var")
amplitude = LogNormal(0., 0.3, name="amplitude")
freq = LogNormal(loc=1., scale=1., name="freq")
mu = ConstantMean(0.5)
cov = SquaredExponential(scale=length_scale)*Harmonic(frequency=freq)*amplitude + WhiteNoise(magnitude=noise_var)
f = GP(mu, cov, name="f")
y = f(x)
model = ProbabilisticModel([y])

# Observe data
noise_level = 0.35 #35
f0 = 1.8 #2
df = 0.
data = np.sin(2*np.pi*(f0 + df*x_range)*x_range) + noise_level*np.random.normal(0., 1., (1, num_datapoints))
y.observe(data)
#plt.scatter(x_range, data.flatten())
#plt.show()


# Variational model
num_particles = 3
initial_locations = [(np.random.normal(0.8, 0.3), np.random.normal(0.8, 0.3),
                      np.random.normal(0.8, 0.3), np.random.normal(0.8, 0.3)) #0.2
                     for _ in range(num_particles)]
particles = [ProbabilisticModel([RootVariable(location[0], name="length_scale", learnable=True),
                                 RootVariable(location[1], name="noise_var", learnable=True),
                                 RootVariable(location[2], name="amplitude", learnable=True),
                                 RootVariable(location[2], name="freq", learnable=True)])
             for location in initial_locations]


# Importance sampling distributions
variational_samplers = [ProbabilisticModel([LogNormal(np.log(location[0]), 0.2, name="length_scale", learnable=True),
                                            LogNormal(np.log(location[1]), 0.2, name="noise_var", learnable=True),
                                            LogNormal(np.log(location[2]), 0.2, name="amplitude", learnable=True),
                                            LogNormal(np.log(location[3]), 0.2, name="freq", learnable=True)])
                        for location in initial_locations]

# Inference
inference_method = WVGD(variational_samplers=variational_samplers,
                        particles=particles,
                        biased=False)
inference.perform_inference(model,
                            inference_method=inference_method,
                            number_iterations=600,
                            number_samples=100,
                            optimizer="SGD",
                            lr=0.001,
                            posterior_model=particles,
                            pretraining_iterations=0)
loss_list = model.diagnostics["loss curve"]
plt.plot(loss_list)
plt.show()

print(inference_method.weights)

final_ensemble = Ensemble(variational_samplers, inference_method.weights)

# Plot samples
colors = ["r", "g", "b"]
M = 20
N = 1000
time_range = np.linspace(-1,3,N)
time_grid_x, time_grid_y = np.meshgrid(x_range, x_range)
diff_grid = time_grid_x - time_grid_y
out_grid_x, out_grid_y = np.meshgrid(time_range, x_range)
out_diff_grid = out_grid_x - out_grid_y
ext_grid_x, ext_grid_y = np.meshgrid(time_range, time_range)
ext_diff_grid = ext_grid_x - ext_grid_y
for idx, sampler in enumerate(variational_samplers):
    ens_mean_f = 0
    ens_mom2_f = 0
    for _ in range(M):
        a = sampler.get_sample(1)
        C = np.matrix(float(a["amplitude"])*np.exp(-diff_grid**2/(2*float(a["length_scale"])**2))*np.cos(2*np.pi*float(a["freq"])*diff_grid) + float(a["noise_var"])*np.identity(num_datapoints))
        out_C = np.matrix(float(a["amplitude"])*np.exp(-out_diff_grid**2/(2*float(a["length_scale"]) ** 2)) * np.cos(2*np.pi*float(a["freq"])*out_diff_grid))
        ext_C = np.matrix(float(a["amplitude"]) * np.exp(-ext_diff_grid ** 2 / (2 * float(a["length_scale"]) ** 2)) * np.cos(2 * np.pi * float(a["freq"]) * ext_diff_grid))
        C_inv = np.linalg.inv(C)
        mean_f = out_C.T*C_inv*np.reshape(data, newshape=(num_datapoints, 1))
        cov_f = ext_C - out_C.T*C_inv*out_C
        var_f = np.diag(cov_f)
        ens_mean_f += np.array(mean_f).flatten()
        ens_mom2_f += np.array(var_f).flatten() -np.array(mean_f).flatten()**2
    ens_mean_f = ens_mean_f/float(M)
    ens_var_f = ens_mom2_f/float(M) + ens_mean_f**2
    plt.plot(time_range, ens_mean_f, alpha=inference_method.weights[idx], label="Particle {}".format(idx), color=colors[idx])
    plt.fill_between(time_range, ens_mean_f - np.sqrt(ens_var_f), ens_mean_f + np.sqrt(ens_var_f), alpha=0.1+2*inference_method.weights[idx]/3., color=colors[idx])
plt.scatter(x_range, data.flatten(), color="k")
plt.xlim(min(time_range), max(time_range))
plt.legend(loc="best")
plt.savefig("predictions.pdf")
plt.show()

# Plot function
import seaborn as sns
def plot_den(model, variables, number_samples=1000):
    sample = model.get_sample(number_samples)
    g = sns.PairGrid(sample[variables])
    g = g.map_offdiag(sns.kdeplot)
    g = g.map_diag(sns.kdeplot, lw=3, shade=True)
    g = g.add_legend()
    g.set(xlim=(0, 3), ylim=(0, 3))

# Plot partition
from brancher.visualizations import plot_particles
plot_particles(particles, var_name="freq", var2_name="noise_var", xlim=(0, 2.5), ylim=(0, 1.5), colors=colors)
ens_samples = final_ensemble.get_sample(20000)
sns.kdeplot(ens_samples["freq"], ens_samples["noise_var"], bw=.2, n_levels=30, cmap="Purples_d")
plt.savefig("partition1.pdf")
plt.show()

# Plot partition
from brancher.visualizations import plot_particles
plot_particles(particles, var_name="amplitude", var2_name="noise_var", xlim=(0, 2.5), ylim=(0, 1.5), colors=colors)
sns.kdeplot(ens_samples["amplitude"], ens_samples["noise_var"], bw=.2, n_levels=30, cmap="Purples_d")
plt.savefig("partition2.pdf")
plt.show()

# Plot partition
from brancher.visualizations import plot_particles
plot_particles(particles, var_name="freq", var2_name="length_scale", xlim=(0, 2.5), ylim=(0, 2.5), colors=colors)
sns.kdeplot(ens_samples["freq"], ens_samples["length_scale"], bw=.2, n_levels=30, cmap="Purples_d")
plt.savefig("partition3.pdf")
plt.show()

#plot_den(final_ensemble, ["freq", "noise_var"])
#plt.savefig("density.pdf")
#plt.show()

# Posterior plot
#plot_den(final_ensemble, ["length_scale", "noise_var", "amplitude", "freq"], number_samples=8000)
#plt.show()

