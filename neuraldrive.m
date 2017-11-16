function neuraldrive
[drawables_train, labels_train, Xs_train, ys_train] = readdata('mnist_train.csv');
[~, labels_test, Xs_test, ~] = readdata('mnist_train.csv');
disp('DATA READ');

roll = floor(rand() * 6000) + 1;
colormap(gray);
imagesc(drawables_train(:, :, roll));
axis off
title(labels_train(roll));

batchsize = 2;
eta = 1;
hiddensize = 64;

W1 = rand(hiddensize, 784) * 2 - 1;
W2 = rand(hiddensize, hiddensize) * 2 - 1;
W3 = rand(10, hiddensize) * 2 - 1;

X = Xs_train(:, :, roll);
[guess, y_] = evaluate(X, W1, W2, W3);
disp(guess);
disp(y_');

% [W1, W2, W3] = sgd(Xs_train, ys_train, batchsize, W1, W2, W3, eta);
% disp(score(W1, W2, W3, Xs_test, labels_test, 6000));

[W1, W2, W3, scores] = scoredsgd(Xs_train, ys_train, batchsize, W1, W2, W3, eta, Xs_test, labels_test, 50, 50);
figure();
plot(scores);
title('Score: ' + string(score(W1, W2, W3, Xs_test, labels_test, 6000)));

[guess, y_] = evaluate(X, W1, W2, W3);

disp(guess);
disp(y_');
end