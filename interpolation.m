%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function creates interpolated PM2.5 levels for the times at 5-minute
% intervals between 11:30AM and 12:25PM on a given day based on the
% available PM2.5 data for the three days surrounding that period. 
% INPUTS:
% "train_data" is a P-by-7 table whose seven columns have the same entries
% as the columns of the N-by-7 matrix provided in the data files: 1) time, 
% 2) humidity, 3) temperature, 4) vehicle speed, 5) pm2d5, 6) latitude, and
% 7) longitude. 
% "test_data" is a M-by-6 table. Its six columns are time, humidity, 
% temperature, vehicle speed, latitude, and longitude.
% OUTPUT:
% "data_by_location" is a h-by-2 cell array where h is the number of test
% locations. The first column contains the prepared training data for each
% location and the second column contains the prepared test data.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function pred_pm2d5 = interpolation(train_data, test_data)
    % find the datetime at which the data begins
    start_datetime=min(train_data.time);

    % partition the data based on datetime in order to get data for the
    % periods only before or after the testing period
    time_cutoffs=start_datetime+duration([35;36],[30;25],[0;0]);
    forward_training=train_data(train_data.time<=time_cutoffs(1),:);
    backward_training=train_data(train_data.time>time_cutoffs(1),:);
    forward_data_by_location = prepare_data(forward_training, ...
        test_data);
    backward_data_by_location = prepare_data(backward_training, ...
        test_data);
    pred_pm2d5=[];
    for loc=1:size(forward_data_by_location,1)
        % train the model and make predictions for the forward problem
        [net, mu, sigma,XTrain] = train_model(forward_data_by_location...
            {loc,1});
        forward_pred_pm2d5=get_forecast(XTrain,...
            forward_data_by_location{loc,2}, net, mu, sigma, 1);
        for i=1:2
            backward_data_by_location{loc,i}.time=...
                -backward_data_by_location{loc,i}.time;
            backward_data_by_location{loc,i}=...
                flip(backward_data_by_location{loc,i});
        end
        % train the model and make predictions for the backward problem
        [net, mu, sigma,XTrain] = train_model(backward_data_by_location...
            {loc,1});
        backward_pred_pm2d5=get_forecast(XTrain,...
            backward_data_by_location{loc,2}, net, mu, sigma, 1);

        % calculate weights
        forward_weights=linspace(1,0,12)';

        % make predictions combining the forward and backward predictions
        combined_pred=forward_pred_pm2d5.*forward_weights+...
            flip(backward_pred_pm2d5).*(1-forward_weights);
        pred_pm2d5=[pred_pm2d5; combined_pred];
    end
end