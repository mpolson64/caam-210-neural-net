function [W1, W2, W3] = trainominibatch(Xs, ys, W1, W2, W3, eta)
W1sum = 0;
W2sum = 0;
W3sum = 0;

for i = 1:size(Xs, 3)
    [dJdW1, dJdW2, dJdW3] = backprop(Xs(:, :, i), ys(:, :, i), W1, W2, W3);
    W1sum = W1sum + dJdW1;
    W2sum = W2sum + dJdW2;
    W3sum = W3sum + dJdW3;
end

W1nudge = W1sum / size(Xs, 3);
W2nudge = W2sum / size(Xs, 3);
W3nudge = W3sum / size(Xs, 3);

W1 = W1 - eta * W1nudge;
W2 = W2 - eta * W2nudge;
W3 = W3 - eta * W3nudge;
end