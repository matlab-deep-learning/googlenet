function net = assembleGoogLeNet()
% assembleGoogLeNet   Assemble GoogLeNet network
%
% net = assembleGoogLeNet creates a GoogLeNet network with weights
% trained on ImageNet. You can load the same GoogLeNet network by
% installing the Deep Learning Toolbox Model for GoogLeNet Network
% support package from the Add-On Explorer and then using the googlenet
% function.

%   Copyright 2019 The MathWorks, Inc.

% Download the network parameters. If these have already been downloaded,
% this step will be skipped.
%
% The files will be downloaded to a file "googlenetParams.mat", in a
% directory "GoogLeNet" located in the system's temporary directory.
dataDir = fullfile(tempdir, "GoogLeNet");
paramFile = fullfile(dataDir, "googlenetParams.mat");
downloadUrl = "http://www.mathworks.com/supportfiles/nnet/data/networks/googlenetParams.mat";

if ~exist(dataDir, "dir")
    mkdir(dataDir);
end

if ~exist(paramFile, "file")
    disp("Downloading pretrained parameters file (26 MB).")
    disp("This may take several minutes...");
    websave(paramFile, downloadUrl);
    disp("Download finished.");
else
    disp("Skipping download, parameter file already exists.");
end

% Load the network parameters from the file googlenetParams.mat.
s = load(paramFile);
params = s.params;

% Create a layer graph with the network architecture of GoogLeNet.
lgraph = googlenetLayers;

% Create a cell array containing the layer names.
layerNames = {lgraph.Layers(:).Name}';

% Loop over layers and add parameters.
for i = 1:numel(layerNames)
    name = layerNames{i};
    idx = strcmp(layerNames,name);
    layer = lgraph.Layers(idx);
    
    % Assign layer parameters.
    pname = replace(name,'-','_');
    layerParams = params.(pname);
    if ~isempty(layerParams)
        paramNames = fields(layerParams);
        for j = 1:numel(paramNames)
            layer.(paramNames{j}) = layerParams.(paramNames{j});
        end
        
        % Add layer into layer graph.
        lgraph = replaceLayer(lgraph,name,layer);
    end
end

% Assemble the network.
net = assembleNetwork(lgraph);

end