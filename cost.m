%function cost determines the cost, or error between the expected and calculated
%     values of the output layer matrix. 
%Inputs: y_ is the calculated output layer matrix using the neural network. y is the
%     actual valued matrix the output layer should look like.
%Output: J is the cost of the expected and calculated values of the output layer matrix.
%     For purposes making the calculus of the Neural Network easier, the cost function
%     performs half squared error, instead of more traditional mean squared error. 
function J = cost(y_, y)
J = 0.5 * sum((y - y_) .^ 2);
end
