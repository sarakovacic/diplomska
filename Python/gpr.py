import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel as WK,\
ExpSineSquared as ESS, RationalQuadratic as RQ, Matern as M



# Specify a GP prior
kernel = 1 * RBF(length_scale = 1)
gp = GPR(kernel = kernel, optimizer = None)
print("Initial Kernel\n%s" % kernel)
X_test = np.array(np.linspace(-9, 5, 1000), ndmin = 2).T
f_mean, f_var = gp.predict(X_test, return_std=True)


# Create a figure
fig_prior = plt.figure(figsize = (20,12))
plt.rcParams.update({'font.size': 20})


# Draw a mean function and 95% confidence interval
plt.plot(X_test, f_mean, 'b-', label='mean function')
upper_bound = f_mean + 1.96 * f_var
lower_bound = f_mean - 1.96 * f_var
plt.fill_between(X_test.ravel(), lower_bound, upper_bound, color = 'b', alpha = 0.1,
                 label='95% confidence interval')

# Draw samples from the posterior and plot
X_samples = np.array(np.linspace(-9, 5, 30), ndmin = 2).T
seed = np.random.randint(10) # random seed
plt.plot(X_samples, gp.sample_y(X_samples, n_samples = 5, random_state = seed), ':')

# Aesthetics
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.xlim(X_test.min(), X_test.max())
plt.ylim(-3, 3)
plt.legend(loc='upper left')
plt.title('Samples(Functions) from a GP prior')
plt.show()
#Initial Kernel
1**2 * RBF(length_scale=1)
