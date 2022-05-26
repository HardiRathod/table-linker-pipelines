To replicate the baseline results, either use already created candidates (Present in AWS + Google drive: https://drive.google.com/drive/folders/1-9FSu6uyB260WexzFJboJSgLGmIjafsC?usp=sharing) and the feature files or recreate new ones. 
Run the string similarity commands after the candidate generation in order to get the string similarity features or follow the feature_1 function from the Table_linker_Pipeline.Or skip this by using already generated data.
Run the pseudo_gt_model_training to train the pseudo model or use the existing ones present in aws or google drive (linked above)
Based on preferences use data processing and model generation in train_leave_one_out.py
Use predictions_and_evaluations in order to understand your model results

Extras:
Always set your experiment_name before running these scripts in order to have everything at one place. If you need to recreate candidate files, download source files and canonicalize them before running the candidate generation. 