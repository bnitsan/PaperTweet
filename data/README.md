# Data

Here we describe the different data files.

## Main files
1) full_data.csv - contains all the data from the Twitter scan under data_pull/scan_twitter.ipynb
2) data_mid.csv  - a subset of full_data.csv that contains only data with Tweet character number in [250,2500].

## Training files
1) data_mid_train.csv, data_mid_val.csv, data_mid_test.csv - a 80/10/10 split of data_mid.csv
2) data_mid_debug_xxxx.csv, like the previous set, but with 10% of the data, for debugging purposes.

## Other
1) full_data_subjects.csv - like full_data.csv, with another column for the subject of the tweet (added in post-processing)
2) custom_data_my_papers.csv - a dataset of my papers, for testing purposes. Can be replaced by other papers using data_pull/Generate_custom_data_from_arxiv.ipynb

