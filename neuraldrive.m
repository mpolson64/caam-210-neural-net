function neuraldrive
[drawables_train, labels_train, Xs_train, ys_train] = readdata('mnist_train.csv');
[drawables_test, labels_test, Xs_test, ys_test] = readdata('mnist_train.csv');
disp('DATA READ');

roll = floor(rand() * 6000);
colormap(gray);
axis off
imagesc(drawables_train(:, :, roll));
title(labels_train(roll));

eta = 1;

W1 = rand(16, 784) * 2 - 1;
W2 = rand(16, 16) * 2 - 1;
W3 = rand(10, 16) * 2 - 1;

disp(score(W1, W2, W3, Xs_test, labels_test));

X = Xs_train(:, :, roll);
y = ys_train(:, :, roll);
[guess, y_] = evaluate(X, W1, W2, W3);
disp(guess);
disp(y_');

[W1, W2, W3] = sgd(Xs_train, ys_train, 1, W1, W2, W3, eta);

disp(score(W1, W2, W3, Xs_test, labels_test));

[guess, y_] = evaluate(X, W1, W2, W3);

disp(guess);
disp(y_');
end

function [guess, y_, a2, a3, z2, z3, z4] = evaluate(X, W1, W2, W3)
z2 = feedforward(X, W1);
a2 = arrayfun(@(x) sigmoid(x), z2);
z3 = feedforward(a2, W2);
a3 = arrayfun(@(x) sigmoid(x), z3);
z4 = feedforward(a3, W3);
y_ = arrayfun(@(x) sigmoid(x), z4);

guess = ytolabel(y_);
end

function z = feedforward(a, W)
z = W * a;
end

function [dJdW1, dJdW2, dJdW3] = backprop(X, y, W1, W2, W3)
[~, y_, a2, a3, z2, z3, z4] = evaluate(X, W1, W2, W3);

delta4 = (-1 * (y - y_)) .* arrayfun(@(x) sigmoidprime(x), z4);
dJdW3 = delta4 * a3';

delta3 = W3' * delta4 .* arrayfun(@(x) sigmoidprime(x), z3);
dJdW2 = delta3 * a2';

delta2 = W2' * delta3 .* arrayfun(@(x) sigmoidprime(x), z2);
dJdW1 = delta2 * X';
end

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

function [W1, W2, W3] = sgd(Xs, ys, batchsize, W1, W2, W3, eta)
for i = 1:(floor(size(Xs, 3) / batchsize) - batchsize)
    Xbatch = Xs(:, :, batchsize * i + (1:batchsize));
    ybatch = ys(:, :, batchsize * i + (1:batchsize));
    
    [W1, W2, W3] = trainominibatch(Xbatch, ybatch, W1, W2, W3, eta);
end
end

function y = sigmoid(x)
y = 1 / (exp(-x) + 1);
end

function y = sigmoidprime(x)
y = exp(-x) / ((1 + exp(-x)) ^ 2);
end

function J = cost(y_, y)
J = 0.5 * sum((y - y_) .^ 2);
end

function val = score(W1, W2, W3, Xs, labels)
correct = 0;
for i = 1:size(Xs, 3)
    guess = evaluate(Xs(:, :, i), W1, W2, W3);
    
    if guess == labels(i)
        correct = correct + 1;
    end
end

val = correct / size(Xs, 3);
end

function [drawables, labels, inputs, expectedvalues] = readdata(filename)
trainingdata = csvread(filename);

drawables = zeros(28, 28, 6000);
for i = 1:6000
    drawables(:, :, i) = reshape(trainingdata(i, 2:end), [28, 28])';
end

labels = trainingdata(:, 1);

inputs = zeros(784, 1, 6000);
for i = 1:6000
    inputs(:, :, i) = trainingdata(i, 2:end)' ./ 255;
end

expectedvalues = zeros(10, 1, 6000);
for i = 1:6000
   expectedvalues(labels(i) + 1, 1, i) = 1;
end
end

function label = ytolabel(y)
label = find(y == max(y)) - 1;
end