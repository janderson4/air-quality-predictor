%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PM2.5 Predictor Function
% CEE254 Autumn 2024
% GROUP 7
%
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
% "pred_pm2d5", which is a M-by-1 vector corresponding to each problem
% type. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function pred_pm2d5 = pm2d5_pred_model(train_data, test_data, problem_type)
    switch problem_type
        case {1,2} % short-term prediction
            data_by_location = prepare_data(train_data, test_data);
            pred_pm2d5=[];
            for loc=1:size(data_by_location,1)
                [net, mu, sigma,XTrain] = train_model(data_by_location...
                    {loc,1});
                pred_pm2d5=[pred_pm2d5; get_forecast(XTrain,...
                    data_by_location{loc,2}, net, mu, sigma, 12)];
            end
        case 3 % interpolation
            pred_pm2d5 = interpolation(train_data, test_data);
    end
end  