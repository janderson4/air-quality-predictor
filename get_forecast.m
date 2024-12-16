%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function uses a trained neural network to make predictions for PM2.5
% levels. 
% INPUTS:
% "XTrain" is the array of training data. 
% "test_data" is a M-by-6 table. Its six columns are time, humidity, 
% temperature, vehicle speed, latitude, and longitude. 
% "net" is the trained neural network. 
% "mu" is the mean PM2.5 level for training data, used for normalization. 
% "sigma" is the mean PM2.5 level for training data, used for 
% normalization. 
% "steps_between_predictions" is the number of steps of the LSTM model that
% should be run between each of the output times. 
% OUTPUT:
% "pred_pm2d5" is a vector containing the predicted PM2.5 level at the test
% times.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function pred_pm2d5 = get_forecast(XTrain, test_data, net, mu, sigma,...
    steps_between_predictions)
    numChannels=size(XTrain,2);
    pm2d5_channel=numChannels;

    % create a matrix for the test data
    XTest=[test_data.time, test_data.hmd, test_data.tmp, ...
        test_data.hour_sin, test_data.hour_cos];

    % update the neural network with the last 200 rows of training data
    sequenceLength = 200;
    XLast = XTrain(end-sequenceLength+1:end,:);
    net = resetState(net);
    % make a prediction for the first testing time step
    [Z, state] = predict(net, XLast);
    net.State = state;
    
    numPredictionTimeSteps = height(test_data)*steps_between_predictions;
    % Y is the matrix containing the data for all channels for the time
    % steps after the last training step
    Y = zeros(numPredictionTimeSteps,numChannels);
    % the pm2d5 value for the first prediction step has already been calculated
    Y(1,pm2d5_channel) = Z(end,pm2d5_channel);
    known_vars=setdiff(1:numChannels,pm2d5_channel);
    
    % normalize the test data
    XTest_normalized=(XTest - mu(known_vars)) ./ sigma(known_vars);

    XTest_normalized=[XTrain(end,known_vars);XTest_normalized];
    for i=1:numChannels-1
        channel=known_vars(i);
        for j=1:height(test_data)
            % calculate a linear array between the previous predicted value and
            % the next value
            interpolated=linspace(XTest_normalized(j,i), ...
                XTest_normalized(j+1,i),steps_between_predictions+1);
            Y((j-1)*steps_between_predictions+1:j*...
                steps_between_predictions,channel)=interpolated(1:...
                steps_between_predictions);
        end
    end

    % conduct closed-loop predictions
    for t = 2:numPredictionTimeSteps
        [predicted_row,state] = predict(net,Y(t-1,:));
        Y(t,pm2d5_channel)=predicted_row(pm2d5_channel);
        net.State = state;
    end
    
    % denormalize the predicted PM2.5 levels
    pred_denormal=Y(:,pm2d5_channel)*sigma(pm2d5_channel)+...
        mu(pm2d5_channel);

    % select only the elements of the PM2.5 prediction vector corresponding
    % to the corresponding time steps in the test data
    pred_pm2d5=pred_denormal(steps_between_predictions*...
        (1:height(test_data)));
end