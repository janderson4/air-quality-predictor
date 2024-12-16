%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function changes the training and test data so that the features are
% the ones chosen for training and prediction. 
% INPUTS:
% "data" is a table with four columns: 1) time, 2) humidity, 3)
% temperature, and 4) PM2.5. 
% "test_pos" is a table containing two columns: "Var1", which is the
% latitude, and "Var2", which is the longitude, both for the test location.
% OUTPUT:
% "data" is a table with six columns: 1) time, 2) humidity, 3)
% temperature, 4) PM2.5, 5) the sin of the hour of day, and 6) the cos of
% the hour of day. The "time" column has also been changed to seconds. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

function data = change_vars(data, start_datetime)
    time = data.time;
    % encode time of day in radians
    time_rad=2 * pi * (hour(time)/24+minute(time)/(24*60));

    % add features for the sin and cosine of time
    hour_sin=sin(time_rad);
    data=addvars(data,hour_sin);
    hour_cos = cos(time_rad);
    data=addvars(data,hour_cos);

    data.time = double(seconds(data.time-start_datetime));
end