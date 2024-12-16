%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function trains a neural network to make predictions for PM2.5
% levels. 
% INPUTS:
% "train_data" is a table containing the features to be used for model
% training, including PM2.5 data. 
% "steps_between_predictions" is the number of steps of the LSTM model that
% should be run between each of the output times. 
% OUTPUT:
% "net" is the trained neural network. 
% "muX" is the mean PM2.5 level for training data, used for normalization. 
% "sigmaX" is the mean PM2.5 level for training data, used for 
% normalization. 
% "XTrain" is the array of training data. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [net, muX, sigmaX,XTrain] = train_model(train_data)
    X=[train_data.time, train_data.hmd, ...
        train_data.tmp,train_data.hour_sin,...
        train_data.hour_cos,train_data.pm2d5, ...
        ];
    % the channel where PM2.5 levels are stored
    pm2d5_channel=size(X,2);
    numChannels=pm2d5_channel;

    % to train the LSTM model, the data must be copied so that the test
    % data is offset from the training data by one time step
    XTrain = X(1:end-1,:);
    TTrain = X(2:end,:);

    % the mean and variance of the data must be calculated for each feature
    % for normalization
    muX = mean(XTrain);
    sigmaX = std(XTrain,0);
    
    muT = mean(TTrain);
    sigmaT = std(TTrain,0);
    
    % normalize all columns
    XTrain = (XTrain - muX) ./ sigmaX;
    TTrain = (TTrain - muT) ./ sigmaT;

    % set up the deep learning model
    layers = [
        sequenceInputLayer(numChannels)
        lstmLayer(128)
        fullyConnectedLayer(numChannels)];
    
    options = trainingOptions("rmsprop", ...
        MaxEpochs=200, ...
        SequencePaddingDirection="left", ...
        Shuffle="every-epoch", ...
        Plots="training-progress", ...
        Verbose=false);

    % conduct model training
    net = trainnet(XTrain,TTrain,layers,"mse",options);
end