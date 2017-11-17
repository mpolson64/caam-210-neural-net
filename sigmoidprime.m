%Function sigmoidprime computes the value of the derivative of the sigmoid function
%     for a given input value.
%Input: x is a numerical value.
%Output: y is the value of the derivative of the sigmoid function based on x. 
function y = sigmoidprime(x)
y = exp(-x) / ((1 + exp(-x)) ^ 2);
end
