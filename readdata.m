%Function readdata reads the training and testing image data sets for the Neural Network.  
%   Furthermore, readdata maninpulates the training and testing data such that it separates the
%   raw data into data that can render into images, the number of the image in the data set,
%   the data that is actually sent into the neural network, and the number the image actually represents.
%Inputs: input filename takes in a file. For the purposes of this project, file name is 
%   a .csv file that contains 6000 training or 6000 test images.
%Outputs: drawtables is an output that sorts all 6000 images into smaller data sets that can be
%   rendered into images of a a given handwritten digit. Each part of drawtables contains
%   a 28x28 pixel image of that digit. labels corresponds to the number in which a given digit appears
%   in the data set (ie labels ranges from 1-6000) inputs converts each image into data that is sent
%   into the Neural Network (ie converts pixel values from RGB to all values between 0 and 1.
%   expectedvalues is a matrix that contains all the actual value each image in the data set
%   represents. 

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
