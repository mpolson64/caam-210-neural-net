%function evaluate evaluates a test image against the Neural Network and returns
%     what the network perceives the image to represent.
%Inputs: x is an image for the network to guess on. For the purposes of this project, x is a 784x1
%     matrix with all element values between 0 and 1. This matrix represents a 28x28 pixel image of 
%     a handwritten digit. W1, W2, and W3 are the trained weights matrices of the Neural Network.
%Outputs: guess is a number 0-9; it is the number the Neural Network perceives the handwritten digit to
%     be of. y_ is the final output layer of the nerual network. It is a 10x1 matrix, with the largest 
%     element value representing the digit the neural network perceives the image to be of. a2, a3, z2,
%     z3, and z4 are helper structures in function evaluate. They are matrices that show the activations of
%     each neural layer before and after compression by the sigmoid function. 
function [guess, y_, a2, a3, z2, z3, z4] = evaluate(X, W1, W2, W3)
z2 = feedforward(X, W1);
a2 = arrayfun(@(x) sigmoid(x), z2); %arrayfun allows a function to be applied to every element of a matrix
z3 = feedforward(a2, W2);
a3 = arrayfun(@(x) sigmoid(x), z3);
z4 = feedforward(a3, W3);
y_ = arrayfun(@(x) sigmoid(x), z4);

guess = ytolabel(y_);
end
