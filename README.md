# Brancher: An Object-Oriented Variational Probabilistic Programming Library

Brancher allows design and train differentiable Bayesian models using stochastic variational inference. Brancher is based on the deep learning framework Chainer. 

## Example code: Autoregressive modeling ##

### Probabilistic model ###
T = 20
driving_noise = 1.
measure_noise = 0.3
x0 = NormalVariable(0., driving_noise, 'x0')
y0 = NormalVariable(x0, measure_noise, 'x0')
b = LogitNormalVariable(0.5, 1., 'b')

x = [x0]
y = [y0]
x_names = ["x0"]
y_names = ["y0"]
for t in range(1,T):
    x_names.append("x{}".format(t))
    y_names.append("y{}".format(t))
    x.append(NormalVariable(b*x[t-1], driving_noise, x_names[t]))
    y.append(NormalVariable(x[t], measure_noise, y_names[t]))
AR_model = ProbabilisticModel(x + y)


### Observe data ###
[yt.observe(data[yt][:, 0, :]) for yt in y]

### Autoregressive variational distribution ###
Qb = LogitNormalVariable(0.5, 0.5, "b", learnable=True)
logit_b_post = DeterministicVariable(0., 'logit_b_post', learnable=True)
Qx = [NormalVariable(0., 1., 'x0', learnable=True)]
Qx_mean = [DeterministicVariable(0., 'x0_mean', learnable=True)]
for t in range(1, T):
    Qx_mean.append(DeterministicVariable(0., x_names[t] + "_mean", learnable=True))
    Qx.append(NormalVariable(BF.sigmoid(logit_b_post)*Qx[t-1] + Qx_mean[t], 1., x_names[t], learnable=True))
variational_posterior = ProbabilisticModel([Qb] + Qx)

### Inference ###
loss_list = inference.stochastic_variational_inference(AR_model, variational_posterior,
                                                       number_iterations=100,
                                                       number_samples=300,
optimizer=chainer.optimizers.Adam(0.05))

















