%% Classify Image Using GoogLeNet
% This example shows how to classify an image using the GoogLeNet
% pretrained convolutional neural network.

%   Copyright 2019 The MathWorks, Inc.

% Read an example image.
img = imread("peppers.png");

% The image that you want to classify must have the same size as the input
% size of the network. Resize the image to be 224-by-224 pixels, the input
% size of GoogLeNet.
img = imresize(img,[224 224]);

% Assemble the pretrained GoogLeNet network. Alternatively, you can
% create a pretrained GoogLeNet network by installing the Deep Learning
% Toolbox Model for GoogLeNet Network support package from the Add-On
% Explorer using the googlenet function.
net = assembleGoogLeNet;

% Analyze the network architecture.
analyzeNetwork(net)

% Classify the image using the network.
label = classify(net,img);

% Display the image together with the predicted label.
figure
imshow(img)
title(string(label))