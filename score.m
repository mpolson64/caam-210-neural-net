function val = score(W1, W2, W3, Xs, labels, samplesize)
correct = 0;
for i = 1:samplesize
    roll = floor(rand() * size(Xs, 3)) + 1; %chooses a random number for the Network to evaluate on
    guess = evaluate(Xs(:, :, roll), W1, W2, W3);
    
    if guess == labels(roll)
        correct = correct + 1;
    end
end

val = correct / samplesize;
end
