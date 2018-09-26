import matplotlib.pyplot as plt
import numpy as np

def gaus(x,mu,sigma):
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2))

x = np.linspace(-3,3,200)
peak = gaus(0.5,0.5,0.5)
x_0 = gaus(0,0.5,0.5)
plt.plot(x,gaus(x,0.5,0.5),'-',label=r"$\mu=0.5,\sigma=0.5$")
plt.plot(x,gaus(x,-0.5,0.5),"--",label=r"$\mu=-0.5, \sigma=0.5$")
plt.plot((0,0),(0,peak),"r:",label="Wahrheit")
plt.plot((0,-0.5),(peak,peak),"g-",label="Verzerrung")
plt.plot((0,0.5),(x_0,x_0),"k-",label="Auflösung")
plt.legend()
plt.xlabel("relative Fehler")
plt.ylabel("Dichtefunktion")
plt.savefig("pictures/Resolution_bias_plot.pdf")
