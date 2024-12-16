%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function groups training data into 5-minute intervals where all
% features are averaged over those intervals. The average for each feature
% is weighted by distance. 
% INPUTS:
% "table1" is a P-by-7 table whose seven columns have the same entries
% as the columns of the N-by-7 matrix provided in the data files: 1) time, 
% 2) humidity, 3) temperature, 4) vehicle speed, 5) pm2d5, 6) latitude, and
% 7) longitude. 
% "test_pos" is a table containing two columns: "Var1", which is the
% latitude, and "Var2", which is the longitude, both for the test location.
% OUTPUT:
% "grouped" is a table with four columns: 1) time, 2) humidity, 3)
% temperature, and 4) PM2.5. These columns have the grouped averages for
% the training data. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function grouped = group(table1, test_pos)
    % replace the datatime of each row in "table1" with the datetime for
    % the corresponding 5-minute interval in which that entry lies
    mins=table1.time.Minute;
    table1.time=table1.time-seconds(table1.time.Second)+...
        minutes(-mins+5*floor(mins/5));

    % an array containing the coordinates of all the training positions
    pos=[table1.lat, table1.lon];

    % the coordinates of the test location
    test_pos=[test_pos.Var1(1), test_pos.Var2(1)];

    n=length(pos);

    % calculate the distance between the test location and each sensor
    % location
    distances=sqrt(sum((repmat(test_pos,n,1)-pos).^2,2));

    % calculate the weights for all training locations by a Gaussian-like
    % function
    table1.weights=exp(-distances.^2);
    varNames=["hmd","tmp","pm2d5","weights"];

    % multiply the features by their weights
    table1.hmd=table1.hmd.*table1.weights;
    table1.tmp=table1.tmp.*table1.weights;
    table1.pm2d5=table1.pm2d5.*table1.weights;

    % define a function for taking the sum of rows matching a condition for
    % a given column in the table
    f_sum = @(x)sum(x,1);

    % group all the rows with the same 5-minute interval
    grouped=grpstats(table1,'time',f_sum,"DataVars",varNames);

    time=grouped.time;
    hmd=grouped.Fun1_hmd./grouped.Fun1_weights;
    tmp=grouped.Fun1_tmp./grouped.Fun1_weights;
    pm2d5=grouped.Fun1_pm2d5./grouped.Fun1_weights;
    grouped=table(time,hmd,tmp,pm2d5);

end