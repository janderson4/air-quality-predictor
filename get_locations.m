function [locations,test_data2]=get_locations(test_data)
    locations=unique(table(round(test_data.lat,6),round(test_data.lon,6)));
    n=height(locations);
    m=height(test_data);
    tests_per_loc=round(m/n);
    test_data2=cell(n,1);
    for i=1:n
        range1=(i-1)*tests_per_loc+1:i*tests_per_loc;
        data_struct=struct( ...
        "time",test_data.time(range1), ...
        "hmd",test_data.hmd(range1),...
        "tmp",test_data.tmp(range1) ...
        );
        test_data2{i}=struct2table(data_struct);
    end
end