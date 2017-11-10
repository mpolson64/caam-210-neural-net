function neuraldrive
[trainingdata, drawables, labels, expectedvalues] = readtraining();

roll = floor(rand() * 6000)
imagesc(drawables(:, :, roll));
title(labels(roll));

weights1 = rand(16, 28 * 28) * 2 - 1;
weights2 = rand(16, 16) * 2 - 1;
weights3 = rand(10, 16) * 2 - 1;
in = trainingdata(1, 2:end)' ./ 255;

mid1 = feedforward(in, weights1);
mid2 = feedforward(mid1, weights2);
out = feedforward(mid2, weights3)

expect = expectedvalues(:, 1, 1)
bad = mse(out, expect)
end
