%Function sigmoid calculates the sigmoid function for a given input value
%Input: x is a numerical value.
%Output: y is the value of the sigmoid function based on x.
function y = sigmoid(x)
y = 1 / (exp(-x) + 1);
end
