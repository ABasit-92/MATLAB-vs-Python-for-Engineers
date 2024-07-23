import numpy as np
import matplotlib.pyplot as plt

mu_min = 2.4    
mu_max = 4 

n_mu = 500 # mu pixels
n_x = 400 # x pixels

mu_edges = np.linspace(mu_min , mu_max, n_mu+1)
mu=(mu_edges[0:n_mu]+mu_edges[1:n_mu+1])/2

# edges of x-pixels
x_edges=np.linspace(0, 1, n_x + 1)

# transient iterations
n_trans = 20000
# number of x-values per mu value
n_data = 10000
# for constructing the figure
x_data= np.zeros((n_data, n_mu)) 

# initial condition

x_0 = 0.5

# for-loop

for i in range(n_mu):
    x = x_0
    
    for j in range(n_trans):
        x = mu[i] * x * (1 - x)
        
    
    for j in range(n_data):
        x = mu[i] * x * (1 - x)
        x_data[j, i] = x
        


# binning the data 

x_histogram = np.zeros((n_x, n_mu))

for i in range(n_mu):
    hist, a = np.histogram(x_data[:, i], bins=x_edges)
    hist = 255 * hist / np.max(hist)
    x_histogram[:, i] = hist


# Plotting the logistic map

plt.figure(figsize=(6, 4))
plt.imshow(x_histogram, extent=[mu_edges[0], mu_edges[-1], x_edges[0], x_edges[-1]], 
           aspect='auto', origin='lower', cmap='gray_r', vmin=0, vmax=255)
plt.xlabel(r'$\mu$', fontsize=14)
plt.ylabel(r'$x$', fontsize=14)
plt.title('Logistic Map Bifurcation Diagram in Python', fontsize=16)
plt.show()