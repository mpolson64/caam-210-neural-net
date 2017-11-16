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