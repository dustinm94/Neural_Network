import numpy as np

# X = (Hours Sleeping, Hours Studying), Y = Test Score

X = np.array(([2, 9], [1, 5], [3, 6]), dtype = float)
Y = np.array(([92], [86], [89]), dtype = float)
xPredicted = np.array(([4,8]))

# Scaling the Data
X = X/np.amax(X, axis = 0) # Max of X
xPredicted = xPredicted / np.amax(xPredicted, axis = 0)
Y = Y/100 # Max Test Score 100

class Neural_Network():
    def __init__(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        # Weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # 3x2 Input -> Hidden
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # 3x1 Hidden -> Output

    def forward(self, X):
        #Forward Propagation
        self.z = np.dot(X, self.W1) # dot product of X and first set of 3x2 weights
        self.z2 = self.sigmoid(self.z) # activation
        self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer and second set of 3x1 weights
        o = self.sigmoid(self.z3)
        return o

    def sigmoid(self, s):
        # activation
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
        # derivative of the sigmoid function
        return s * (1 - s)

    def backward(self, X,Y,o):
        #backward propagation
        self.o_error = Y - o # Error in the output
        self.o_delta = self.o_error * self.sigmoidPrime(o) # applies derivative of sigmoid to error

        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)

        self.W1 = X.T.dot(self.z2_delta) # adjusting first set of input weights input -> hidden
        self.W2 = self.z2.T.dot(self.o_delta) # adjusts second set of hidden weights hidden -> output

    def train(self, X,Y):
        o = self.forward(X)
        self.backward(X,Y,o)

    def saveWeights(self):
        np.savetxt("w1.txt", self.W1, fmt = "%s")
        np.savetxt("w2.txt", self.W2, fmt = "%s")

    def predict(self):
        print("Predicted data based on trained weights: ")
        print("Scaled Input: \n" + str(xPredicted))
        print("Output: \n" + str(self.forward(xPredicted)))

NN = Neural_Network()
o = NN.forward(X)

for i in range(1000): # Help trains the NN 1000 times.
    print("Input: \n" + str(X))
    print("Real Output: \n" + str(Y))
    print("Predicted Outpt: \n" + str(NN.forward(X)))
    print("Loss: \n" + str(np.mean(np.square(Y - NN.forward(X)))))
    print("\n")
    NN.train(X,Y)

NN.predict()





