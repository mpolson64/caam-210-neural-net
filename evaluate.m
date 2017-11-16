function [guess, y_, a2, a3, z2, z3, z4] = evaluate(X, W1, W2, W3)
z2 = feedforward(X, W1);
a2 = arrayfun(@(x) sigmoid(x), z2);
z3 = feedforward(a2, W2);
a3 = arrayfun(@(x) sigmoid(x), z3);
z4 = feedforward(a3, W3);
y_ = arrayfun(@(x) sigmoid(x), z4);

guess = ytolabel(y_);
end