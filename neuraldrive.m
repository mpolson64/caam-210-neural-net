%function neuraldrive is a driver function that combines all necessary functions 
%     to train and score a handwritten digit recognition Neural Network. 
%Input: none (this is a driver)
%Output: none (this is a driver)
%Note: after creating the GUI to demonstrate the neural network, neuraldrive was no longer used.
%     This driver was only used in verifying that the network trained and identified test images successfully.
function neuraldrive
[drawables_train, labels_train, Xs_train, ys_train] = readdata('mnist_train.csv');
[~, labels_test, Xs_test, ~] = readdata('mnist_train.csv');
disp('DATA READ');

roll = floor(rand() * 6000) + 1;
%colormap(gray);
%imagesc(drawables_train(:, :, roll));
%axis off
%title(labels_train(roll));

batchsize = 2;
eta = 1;
hiddensize = 64;

W1 = rand(hiddensize, 784) * 2 - 1;
W2 = rand(hiddensize, hiddensize) * 2 - 1;
W3 = rand(10, hiddensize) * 2 - 1;

X = Xs_train(:, :, roll);
[guess, y_] = evaluate(X, W1, W2, W3);
%disp(guess);
%disp(y_');

% [W1, W2, W3] = sgd(Xs_train, ys_train, batchsize, W1, W2, W3, eta);
% disp(score(W1, W2, W3, Xs_test, labels_test, 6000));

[W1, W2, W3, scores] = scoredsgd(Xs_train, ys_train, batchsize, W1, W2, W3, eta, Xs_test, labels_test, 50, 50);
%figure();
%plot(scores);
%title('Score: ' + string(score(W1, W2, W3, Xs_test, labels_test, 6000)));

[guess, y_] = evaluate(X, W1, W2, W3);

%disp(guess);
%disp(y_');
failsmat = [];
theguess = [];
fails = 0;
Atotal = [];
cnt = 0;
for i = 1:50
    display(cnt)
    W1 = rand(hiddensize, 784) * 2 - 1;
    W2 = rand(hiddensize, hiddensize) * 2 - 1;
    W3 = rand(10, hiddensize) * 2 - 1;
    [W1, W2, W3, scores] = scoredsgd(Xs_train, ys_train, batchsize, W1, W2, W3, eta, Xs_test, labels_test, 50, 50);
    fails = 0;
    failsmat = [];
    theguess = [];
    for i = 1:6000
       X = Xs_train(:,:,i);
       guess = evaluate(X, W1, W2, W3);
       if labels_test(i) ~= guess
           fails = fails + 1;
           failsmat = [failsmat, labels_test(i)];
           theguess = [theguess, guess];
       end
    end
    fails = (fails/6000) * 100;
    display("The failure rate was " + num2str(fails) + "%")
    display("The handwritten digit most incorrectly identified was " + num2str(mode(failsmat)))
    positions = find(failsmat ==mode(failsmat));
    whatitidentifiedas = mode(theguess(positions));
    display("This digit was most commonly guessed to be " + whatitidentifiedas)
    A = [mode(failsmat), whatitidentifiedas, fails];
    Atotal = [Atotal;A];
%     fileID = fopen('TestData.txt','w');
%     fprintf(fileID,'%6.1f %12.1f %12.1\r\n',A)
%     fclose(fileID);
    cnt = cnt+1;
end
display(Atotal)
% fid = fopen('AnotherTest.txt','wt');
% for ii = 1:size(Atotal,1)
%     fprintf(fid,'%g\t',Atotal(ii,:));
%     fprintf(fid,'\n');
% end
% fclose(fid)
csvwrite('Tester2.csv',Atotal)
end
