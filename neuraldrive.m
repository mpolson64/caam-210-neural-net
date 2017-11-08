function neuraldrive
%Read training data and labels
trainingdata = csvread('mnist_train.csv');
digits = zeros(28, 28, 5999);
for i = 1:5999
    digits(:, :, i) = reshape(trainingdata(i, 2:end), [28, 28])';
end
labels = trainingdata(:, 1);
end
