%Function backprop performs the backpropagation algorithm on the weights matrices of a 
%     Neural Network as part of improving the network's chances of digit recognition. To
%     accomplish backpropagation, function backprop uses a modified version of the chain rule from calculus. 
%Inputs: W1, W2, and W3 are the weights matrices of the Neural Network. X and y are inputs
%     that describe an input image from which backpropagation can improve the Network.
%Outputs: djdw1, djdw2, and djdw3 are matrices that tell how much each weights matrix needs to be
%     adjusted for the Neural Network to better regcognize hand written digits. 
function [dJdW1, dJdW2, dJdW3] = backprop(X, y, W1, W2, W3)
[~, y_, a2, a3, z2, z3, z4] = evaluate(X, W1, W2, W3);

delta4 = (-1 * (y - y_)) .* arrayfun(@(x) sigmoidprime(x), z4);
dJdW3 = delta4 * a3';

delta3 = W3' * delta4 .* arrayfun(@(x) sigmoidprime(x), z3);
dJdW2 = delta3 * a2';

delta2 = W2' * delta3 .* arrayfun(@(x) sigmoidprime(x), z2);
dJdW1 = delta2 * X';
end
