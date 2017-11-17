%Function score applies a percentage value to the success rate of a trained weights Neural
%     Network by seeing how often the trained Neural Network correctly identifies images.
%Inputs: W1, W2, and W3 are trained weights matrices. Xs is a matrix of image data to feed
%     into the Neural Network. Labels is a matrix that holds the actual digit values of the images
%     fed into the Neural Network. Samplesize is the number of images the neural network will attempt
%     to correctly identify.
%Output: val is the percentage of sample images the neural network correctly identifies expressed as a decimal.
function val = score(W1, W2, W3, Xs, labels, samplesize)
correct = 0;
for i = 1:samplesize
<<<<<<< HEAD
    roll = floor(rand() * size(Xs, 3)) + 1; %chooses a random number for the Network to evaluate on
=======
    roll = floor(rand() * samplesize) + 1; %chooses a random number for the Network to evaluate on
>>>>>>> thesalmonification-patch-1
    guess = evaluate(Xs(:, :, roll), W1, W2, W3);
    
    if guess == labels(roll)
        correct = correct + 1;
    end
end

val = correct / samplesize;
end
