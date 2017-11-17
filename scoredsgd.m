%Function scoregsd combines the trainominibatch and score functions in one, to measure improvement of the Neural Network under
%     stochastic gradient descent.
%Inputs: Xs_train and ys_train are inputs of the image data for the Network to train upon. W1, W2, and W3 are the weights matrices
%     eta is the learning rate. Xs_test is test images for the Neural Network to be tested against. labels_test are the actual digit
%     values of the test images.
%Outputs: W1, W2, and W3 are the trained weights matrices of the system. scores is a number representing the improvement of the Network
%     upon training. 
function [W1, W2, W3, scores] = scoredsgd(Xs_train, ys_train, batchsize, W1, W2, W3, eta, Xs_test, labels_test, samplesize, res)
scores = zeros(1, floor(batchsize / res));

for i = 1:(floor(size(Xs_train, 3) / batchsize) - batchsize)
    Xbatch = Xs_train(:, :, batchsize * i + (1:batchsize));
    ybatch = ys_train(:, :, batchsize * i + (1:batchsize));
    
    [W1, W2, W3] = trainominibatch(Xbatch, ybatch, W1, W2, W3, eta);
    
    if mod(i, res) == 0
       scores(i / res) = score(W1, W2, W3, Xs_test, labels_test, samplesize);
    end
end
end
