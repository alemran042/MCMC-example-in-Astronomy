import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import matplotlib.pyplot as plt
plt.rc('font', family='serif')
plt.rc('mathtext', fontset='cm')
import pdb

# Read data and assign variables
data = np.loadtxt('dc-data_set.txt')
data = np.transpose(data)
x = data[0]  # x-axis data
y = data[1]  ## y-axis data
yerr = data[2] # error on y-axis

def model(theta, x=x):
	''' Model defined'''
	a, b, c, d, e = theta
	y = (a-b*np.exp(-(x-c)**2/(2*d**2)))*x**(-e)
	return y

def lnlike(theta, x, y, yerr):
	'''Fuction represents how good the model is'''
	LnLike = -0.5*np.sum(((y-model(theta, x))/yerr)**2)
	return LnLike

def lnprior(theta):
	a, b, c, d, e = theta
	if 0. < a < 1000. and 0. < b < 1000.0 and 0. < c < 5. and 0. < d < 5. and 0. < e < 5.:
		return 0.0
	else:
		return -np.inf

def lnprob(theta, x, y, yerr):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, x, y, yerr)

######## Set-up the initials ###########
########################################

data = (x, y, yerr) # Define data array
nwalkers = 240 #  of walker
niter = 5000  # of iteration
initial = np.array([1000., 1000., 2., 2., 2.]) # Set initial value
ndim = len(initial)
step = 1e-2 ## Define step size Stp size
p0 = [np.array(initial) + step * np.random.randn(ndim) for i in range(nwalkers)] 

########## Run the MCMC #############
#####################################

def main(p0,nwalkers,niter,ndim,lnprob,data):
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

	print("Running burn-in...")
	p0, _, _ = sampler.run_mcmc(p0, 100)
	sampler.reset()

	print("Running production...")
	pos, prob, state = sampler.run_mcmc(p0, niter)
	return sampler, pos, prob, state

############ Extract result ############
########################################

sampler, pos, prob, state = main(p0, nwalkers, niter, ndim, lnprob, data)
samples = sampler.flatchain

theta_max  = samples[np.argmax(sampler.flatlnprobability)]
best_fit_model = model(theta_max)
plt.errorbar(x, y, yerr = yerr, marker='o', markersize=8,linestyle='none',mfc='black',mec='black',\
			 capsize= 5, color='black')
plt.plot(x, best_fit_model, 'r',linestyle='solid',label= 'Best fit')

plt.legend(frameon=False, fontsize='large', loc = 'best')
plt.xlabel('X',fontsize=18)
plt.ylabel(r'Y',fontsize=18)

plt.tick_params(which='both',axis='both',direction='in',top=True,right=True,labelsize=14)
plt.tick_params(which='major', length=10)
plt.minorticks_on()
plt.tick_params(which='minor', length=5)
plt.xscale('log')
plt.yscale('log')
plt.savefig('Best_fit_mcmc.png')

print('###############################')
print('Theta max (parameters): ', np.around(theta_max, 2))
print('###############################')

####### Corner plot (median and 16-84% confidence) ###########
##############################################################

labels = [r'A',r'B',r'$x_0$',r'$\sigma_0$', r'$\alpha$']
fig = corner.corner(samples, show_titles=True, labels=labels, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84])
fig.savefig('Corner_plot_mcmc.png')
plt.show()

##################################################
############# Posterior with STD #################

def sample_walkers(nsamples, flattened_chain):
	models = []
	draw = np.floor(np.random.uniform(0, len(flattened_chain), size=nsamples)).astype(int)
	thetas = flattened_chain[draw]
	for i in thetas:
		mod = model(i)
		models.append(mod)
	spread = np.std(models,axis=0)
	med_model = np.median(models,axis=0)
	return med_model,spread

########## Posterior sampling #########
########################################

med_model, spread = sample_walkers(100, samples)

######## Plot the best fit with 1-sigma error ##########
########################################################

plt.errorbar(x, y, yerr = yerr, marker='o', markersize=8,linestyle='none',mfc='black',mec='black',\
			 capsize= 5, color='black')
plt.plot(x, best_fit_model, 'r', label='Highest Likelihood Model')
plt.fill_between(x, med_model-spread, med_model+spread, color='grey', alpha=0.5, label=r'$1\sigma$ Posterior Spread')
plt.xlabel('X',fontsize=18)
plt.ylabel(r'Y',fontsize=18)

plt.tick_params(which='both',axis='both',direction='in',top=True,right=True,labelsize=14)
plt.tick_params(which='major', length=10)
plt.minorticks_on()
plt.tick_params(which='minor', length=5)
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('Posterier_STD_mcmc.png')

plt.show()
pdb.set_trace() ## Stop