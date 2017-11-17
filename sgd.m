%Function sgd performs Stochastic Gradient Descent on the Neural Network as a method
%     of increasing the speed at which the Neural Network trains using backpropagation. 
%Inputs: Xs and ys represent input images for use in stochastic gradient descent. 
%     W1,W2,W3 are the weights matrices that need to be trained. eta is the learning
%     rate.
%Outputs: W1, W2, and W3 are improved weights matrices after training through stochastic gradient
%     descent. These weight matrices are better at indentifying images, if only marginally. 
function [W1, W2, W3] = sgd(Xs, ys, batchsize, W1, W2, W3, eta)
for i = 1:(floor(size(Xs, 3) / batchsize) - batchsize)
    Xbatch = Xs(:, :, batchsize * i + (1:batchsize));
    ybatch = ys(:, :, batchsize * i + (1:batchsize));
    
    [W1, W2, W3] = trainominibatch(Xbatch, ybatch, W1, W2, W3, eta);
end
end
