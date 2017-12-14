# What is a Neural Network?
A collection of neurons connected by synapses. These have three layers, input, hidden and output.



# Neural_Network
This is a feedfoward neural network that tracks the hours we studied and slept before a test, and the output is the test score.
You could use this information to find the most optimal time spent studying/sleeping for the best test score. This Neural Network (NN)
has a single hidden layer and three inputs. 

This performs a dot product with input and weights from the user, then the activation function runs and the weights are adjusted.
You can continuously "train" the network to produce better results. 

Scaling inputs so that you can not have a higher test grade than 100.

# Forward Propagation
X is a three by two matrix, Y is a three by one matrix. Each item in X is multiplied by the weight it was given and added to all the other
results. The activation function that was chosen is a sigmoid function.

# Backward Propagation
This is considered the "learning" of the NN. To do this we apply a loss function, in this case mean sum squared was used. 
To find out which way we need to alter our weights we use gradient descent. 
  1. Find the Margin of Error
  2. Apply the derivative of our sigmoid function to the output error. This gives us the delta output sum
  3. We use the delta output sum from the output error to see how much z2 contributed to the error. 
  4. To calculate the delta output sum for z2 we derive our sigmoid function. (sigmoid prime, similar to step 2)
  5. Adjust the weights by performing a dot product of the input layer with z2 delta output sum. For the second weight perform
     a dot product of the z2 layer and the delta output sum.
     
 This is a basic Neural Network, and my first attempt at understanding and trying to create one. Larger networks have many more
 hidden layers and more inputs as well. You can edit the inputs with what you would like to track or predict. 
 
 
