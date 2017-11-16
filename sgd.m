function [W1, W2, W3] = sgd(Xs, ys, batchsize, W1, W2, W3, eta)
for i = 1:(floor(size(Xs, 3) / batchsize) - batchsize)
    Xbatch = Xs(:, :, batchsize * i + (1:batchsize));
    ybatch = ys(:, :, batchsize * i + (1:batchsize));
    
    [W1, W2, W3] = trainominibatch(Xbatch, ybatch, W1, W2, W3, eta);
end
end