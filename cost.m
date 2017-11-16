function J = cost(y_, y)
J = 0.5 * sum((y - y_) .^ 2);
end