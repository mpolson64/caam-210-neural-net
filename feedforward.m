%Function feedfoward passes the activation levels of a previous layer of the neural network
%     through a weights matrix and returns the activations of the next layer of the neural network.
%Inputs: a is a matrix of the activation levels of a given layer of the neural network. W is the weights matrix
%     a must pass through to determine the activation levels of the next layer of the neural network.
%Output: z is a matrix of the activation levels of the next layer of the neural network. NOTE: these activation
%     levels have not yet passed through the sigmoid function, so they do not truly reprent
%     the activation levels that will be passed on further down the network. 
function z = feedforward(a, W)
z = W * a;
end
