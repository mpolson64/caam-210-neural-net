function [dJdW1, dJdW2, dJdW3] = backprop(X, y, W1, W2, W3)
[~, y_, a2, a3, z2, z3, z4] = evaluate(X, W1, W2, W3);

delta4 = (-1 * (y - y_)) .* arrayfun(@(x) sigmoidprime(x), z4);
dJdW3 = delta4 * a3';

delta3 = W3' * delta4 .* arrayfun(@(x) sigmoidprime(x), z3);
dJdW2 = delta3 * a2';

delta2 = W2' * delta3 .* arrayfun(@(x) sigmoidprime(x), z2);
dJdW1 = delta2 * X';
end