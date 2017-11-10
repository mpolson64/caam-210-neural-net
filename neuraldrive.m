function neuraldrive
[drawables, labels, inputs, expectedvalues] = readtraining();

roll = floor(rand() * 6000);
imagesc(drawables(:, :, roll));
title(labels(roll));

W1 = rand(16, 28 * 28) * 2 - 1;
W2 = rand(16, 16) * 2 - 1;
W3 = rand(10, 16) * 2 - 1;

x = inputs(:, :, roll);

a2 = feedforward(x, W1);
a3 = feedforward(a2, W2);

y_ = feedforward(a3, W3);

y = expectedvalues(:, :, roll);
bad = cost(y_, y);
end
