function [drawables, labels, inputs, expectedvalues] = readtraining()
trainingdata = csvread('mnist_train.csv');

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