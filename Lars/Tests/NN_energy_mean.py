import numpy as np

def sigmoid(x, deriv=False):
    return 1/(1+np.exp(-x))

def derror(x):
    return 2*x*(1-x)


#data
X = np.array([[0.01],
            [0.013],
            [0.012]])


y = np.array([[0.011]])

np.random.seed(1)

#Gewichte

syn0 = 2*np.random.random((1,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1
#syn2 = 2*np.random.random((2,1)) - 1

#training
gamma = 0.5

for i in range(100000):

    l0 = X
    l1 = sigmoid(np.dot(l0,syn0))
    l2 = sigmoid(np.dot(l1,syn1))
    #l3 = sigmoid(np.dot(l2,syn2))

    #l3_error = y - l3
    l2_error = y - l2

    if(i%10000 == 0):
        print("Error: ", np.mean(np.abs(l2_error)))

    #l3_delta = l3_error * gamma * derror(l3)

    #l2_error  = l3_delta.dot(syn2.T)

    l2_delta = l2_error * gamma * derror(l2)

    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error * gamma * derror(l1)

    #syn2 += l2.T.dot(l3_delta)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print("Output after training", l2)
