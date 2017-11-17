%Function ytolabel determines the corresponding digit 0-9 of a final layer output matrix by
%    finding the maximum value within the final layer matrix and subtracting 1.
%Input: y is a final layer output matrix. For the purposes of this project, y will 
%     always be a 10x1 matrix that corresponds to the digits 0-9.
%Output: label is the digit 0-9 that corresponds to y.
function label = ytolabel(y)
label = find(y == max(y)) - 1;
end
