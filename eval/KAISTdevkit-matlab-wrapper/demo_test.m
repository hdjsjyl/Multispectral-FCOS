function demo_test()

%% specify path of detections, STORE NEW DETECTIONS UNDER "det_cus" FOLDER ONLY, DELETE "des_cus.txt" files from "det" folder
dtDir = 'PATH_TO/det';

%% specify path of groundtruth annotaions
gtDir = 'PATH_TO/GROUND_TRUTH';

%% evaluate detection results
kaist_eval_full(dtDir, gtDir, false, true);

end
