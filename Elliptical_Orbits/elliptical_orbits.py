import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Given Conditions
period = 1
omega =  2*np.pi/period

e = np.array([0, 1/4, 1/2, 3/4])
a = 1./(1-e)
b = np.sqrt((1+e)/(1-e))

color = ['r', 'g', 'b', 'c']

t=np.linspace(0,period,1000)

x = np.zeros((len(t), len(e)))
y = np.zeros((len(t), len(e)))


# Defining Functions
def func(E, e, omega, t):
    return E - e * np.sin(E) - omega * t

def orbits(e, omega, t):
    a = 1 / (1 - e)
    b = np.sqrt((1 + e) / (1 - e))

    E = fsolve(func, 0, args=(e, omega, t))

    x = a * (e - np.cos(E))
    y = b * np.sin(E)

    return x,y 

# Looping through and getting the coordinates for the orbits
for j in range(len(e)):
  for i in range(len(t)):
    x[i, j], y[i, j] = orbits(e[j], omega, t[i])


# plotting the orbit
for j in range(len(e)):
  plt.plot(x[:, j], y[:, j], color[j], linewidth=2)


plt.plot(0,0,"x",color='k')

# Decorating the Plot
plt.axis('equal')
plt.legend(["e=0","e=1/4","e=1/2", "e=3/4"], loc ="upper right")
plt.title("Planetary Orbits")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()

# Showing the Plot, Voila !
plt.show()