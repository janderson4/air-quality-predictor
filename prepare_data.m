%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function takes the raw test and train data and prepares it for 
% training the deep learning model. 
% INPUTS:
% "train_data" is a P-by-7 table whose seven columns have the same entries
% as the columns of the N-by-7 matrix provided in the data files: 1) time, 
% 2) humidity, 3) temperature, 4) vehicle speed, 5) pm2d5, 6) latitude, and
% 7) longitude. 
% "test_data" is a M-by-6 table. Its six columns are time, humidity, 
% temperature, vehicle speed, latitude, and longitude. 
% "problem_type" is 1, 2, or 3 for short-term prediction, long-term 
% prediction, or interpolation, respectively. 
% OUTPUT:
% "data_by_location" is a h-by-2 cell array where h is the number of test
% locations. The first column contains the prepared training data for each
% location and the second column contains the prepared test data.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function data_by_location = prepare_data(train_data, test_data)
    % find the datetime at which the data begins
    start_datetime=min(train_data.time);

    % find the locations represented in the test data
    [test_locations,test_data]=get_locations(test_data);

    h=height(test_locations);
    data_by_location=cell(h,2);
    for i=1:h
        % the training data has readings at 3-second intervals for various
        % locations, so it must be grouped into 5-minute averages for the
        % test locations
        train_data_grouped=group(train_data,test_locations(i,:));

        % remove and add features to prepare the model for training
        data_by_location{i,1}=change_vars(train_data_grouped,...
            start_datetime);
        data_by_location{i,2}=change_vars(test_data{i},start_datetime);
    end
end