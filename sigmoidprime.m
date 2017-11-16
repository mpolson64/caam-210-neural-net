function y = sigmoidprime(x)
y = exp(-x) / ((1 + exp(-x)) ^ 2);
end