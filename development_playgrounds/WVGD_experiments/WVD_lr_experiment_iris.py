import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

from brancher.variables import DeterministicVariable, ProbabilisticModel
from brancher.standard_variables import NormalVariable, CategoricalVariable, EmpiricalVariable, RandomIndices
import brancher.functions as BF

from brancher import inference
from brancher.inference import WassersteinVariationalGradientDescent as WVGD

#TODO: Number of particles interface: Work in progress

num_repetitions = 10 #50
particle_numbers = [3, 1] #[10, 8, 6, 4, 2, 1]
results = []
errors = []
post_sample_size = 300

for N in particle_numbers:
    current_results = []
    for R in range(num_repetitions):
        # Data
        number_regressors = 4
        number_output_classes = 3
        dataset_size = 10
        dataset = datasets.load_iris()
        ind = list(range(dataset["target"].shape[0]))
        np.random.shuffle(ind)
        input_variable = dataset["data"][ind[:dataset_size], :].astype("float32")
        output_labels = dataset["target"][ind[:dataset_size]].astype("int32")

        # Data sampling model
        minibatch_size = dataset_size
        minibatch_indices = RandomIndices(dataset_size=dataset_size, batch_size=minibatch_size, name="indices", is_observed=True)
        x = EmpiricalVariable(input_variable, indices=minibatch_indices, name="x", is_observed=True)
        labels = EmpiricalVariable(output_labels, indices=minibatch_indices, name="labels", is_observed=True)

        # Architecture parameters
        weights = NormalVariable(np.zeros((number_output_classes, number_regressors)),
                                 np.ones((number_output_classes, number_regressors)), "weights")

        # Forward pass
        final_activations = BF.matmul(weights, x)
        k = CategoricalVariable(softmax_p=final_activations, name="k")

        # Probabilistic model
        model = ProbabilisticModel([k])

        # Observations
        k.observe(labels)

        # Variational model
        num_particles = N
        initial_locations = [np.random.normal(0., 1., (number_output_classes, number_regressors))
                             for _ in range(num_particles)]
        particles = [ProbabilisticModel([DeterministicVariable(location, name="weights", learnable=True)])
                     for location in initial_locations]

        # Importance sampling distributions
        variational_samplers = [ProbabilisticModel([NormalVariable(loc=location, scale=0.001,
                                                                   name="weights", learnable=True)])
                                for location in initial_locations]

        # Inference
        inference_method = WVGD(variational_samplers=variational_samplers,
                                particles=particles,
                                biased=False)
        inference.perform_inference(model,
                                    inference_method=inference_method,
                                    number_iterations=3000,
                                    number_samples=100,
                                    optimizer="SGD",
                                    lr=0.0025,
                                    posterior_model=particles,
                                    pretraining_iterations=0)
        loss_list = model.diagnostics["loss curve"]

        # Test accuracy
        test_size =  len(ind[dataset_size:])
        num_images = test_size*3
        test_indices = RandomIndices(dataset_size=test_size, batch_size=1, name="test_indices", is_observed=True)
        test_images = EmpiricalVariable(dataset["data"][ind[dataset_size:], :].astype("float32"),
                                        indices=test_indices, name="x_test", is_observed=True)
        test_labels = EmpiricalVariable(dataset["target"][ind[dataset_size:]].astype("int32"),
                                        indices=test_indices, name="labels", is_observed=True)
        test_model = ProbabilisticModel([test_images, test_labels])


        for model_index in range(num_particles):
            s = 0
            log_lk = 0
            model.set_posterior_model(inference_method.sampler_model[model_index])
            scores_0 = []
            test_image_list = []
            test_label_list = []
            for _ in range(num_images):
                test_sample = test_model._get_sample(1)
                test_image, test_label = test_sample[test_images], test_sample[test_labels]
                test_image_list.append(test_image)
                test_label_list.append(test_label)

            for test_image, test_label in zip(test_image_list,test_label_list):
                model_output = np.reshape(np.mean(model._get_posterior_sample(post_sample_size, input_values={x: test_image})[k].detach().numpy(), axis=0), newshape=(number_output_classes,))
                output_label = int(np.argmax(model_output))
                true_label = int(test_label.detach().numpy())
                scores_0.append(1 if output_label == true_label else 0)
                s += 1 if output_label == int(test_label.detach().numpy()) else 0
                log_lk += np.log(float(model_output[true_label]) + 0.000001)
            print("Accuracy {}: {}%, Log likelihood: {}, weight: {}".format(model_index, 100*s/float(num_images), log_lk, inference_method.weights[model_index]))

        s = 0
        scores_ne = []
        for test_image, test_label in zip(test_image_list, test_label_list):
            model_output_list = []
            for model_index in range(num_particles):
                model.set_posterior_model(inference_method.sampler_model[model_index])
                model_output_list.append(np.reshape(np.mean(model._get_posterior_sample(post_sample_size, input_values={x: test_image})[k].detach().numpy(), axis=0), newshape=(number_output_classes,)))

            model_output = sum([output*w for output, w in zip(model_output_list, inference_method.weights)])

            output_label = int(np.argmax(model_output))
            scores_ne.append(1 if output_label == int(test_label.detach().numpy()) else 0)
            s += 1 if output_label == int(test_label.detach().numpy()) else 0
        print("Accuracy Ensemble: {} %".format(100*s/float(num_images)))

        current_results.append(100*s/float(num_images))
    print("Exp {}: {} +- {}".format(N, np.mean(current_results), np.sqrt(np.var(current_results) / num_repetitions)))
    results.append(current_results)
    errors.append((np.mean(current_results), np.sqrt(np.var(current_results)/num_repetitions)))
b = 1

mean, sem = zip(*errors)
mean, sem = np.array(mean), np.array(sem)
plt.scatter(particle_numbers, mean, color="k", lw=2)
plt.plot(particle_numbers, mean, color="k", lw=1)
plt.fill_between(particle_numbers, mean - sem, mean + sem, color="b", alpha=0.5)
plt.savefig("iris_results_WCGD.pdf")
plt.show()
