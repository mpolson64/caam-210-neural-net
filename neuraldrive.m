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
end
