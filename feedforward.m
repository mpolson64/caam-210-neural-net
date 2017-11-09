function outlayer = feedforward(inlayer, weights)
outlayer = arrayfun(@(x) sigmoid(x), weights * inlayer);
end