function outlayer = feedforward(inlayer, weights)
outlayer = 1 - arrayfun(@(x) sigmf(x, [-1, 0]), weights * inlayer);
end