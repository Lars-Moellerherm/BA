import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gerade(x,m,b):
    return m*x+b

# erste Ellipse

x1=1
y1=2
psi1=np.pi/5

#zweite Ellipse

x2=3
y2=3
psi2=np.pi/2

#dritte Ellipse

x3=4
y3=5
psi3=np.pi/7


#Berechnen der Geraden

m1 = y1-np.tan(psi1)*x1/(x1+np.tan(psi1)*y1)
b1 = y1 + np.tan(psi1)*x1**2/(x1+np.tan(psi1)*y1) - x1*y1/(x1+np.tan(psi1)*y1)

m2 = y2-np.tan(psi2)*x2/(x2+np.tan(psi2)*y2)
b2 = y2 + np.tan(psi2)*x2**2/(x2+np.tan(psi2)*y2) - x2*y2/(x2+np.tan(psi2)*y2)

m3 = y3-np.tan(psi3)*x3/(x3+np.tan(psi3)*y3)
b3 = y3 + np.tan(psi3)*x3**2/(x3+np.tan(psi3)*y3) - x3*y3/(x3+np.tan(psi3)*y3)

x=np.linspace(-3,3)
plt.plot(x,gerade(x,m1,b1),'b-',label="(1,2);pi/5")
plt.plot(1,2,'bx')
plt.plot(x,gerade(x,m2,b2),'g-',label="(3,3);pi/2")
plt.plot(3,3,'gx')
plt.plot(x,gerade(x,m3,b3),'k-',label="(4,5);pi/7")
plt.plot(4,5,'kx')
plt.show()
