function J = cost(y_, y)
%Squared error
J = 0.5 * sum((y - y_) .^ 2);
end
