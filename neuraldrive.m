function neuraldrive
%Read training data and labels
trainingdata = csvread('mnist_train.csv');
digits = zeros(28, 28, 6000);
for i = 1:6000
    digits(:, :, i) = reshape(trainingdata(i, 2:end), [28, 28])';
end

labels = trainingdata(:, 1);
expectedvalues = zeros(10, 1, 6000);
for i = 1:6000
   expectedvalues(labels(i) + 1, 1, i) = 1;
end

imagesc(digits(:, :, 88));

weights1 = rand(16, 28 * 28) * 2 - 1
weights2 = rand(10, 16) * 2 - 1
in = trainingdata(1, 2:end)';

mid = feedforward(in, weights1)
out = feedforward(mid, weights2)

expect = expectedvalues(:, 1, 1)
bad = mse(out, expect)
end
