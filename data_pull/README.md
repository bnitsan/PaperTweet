# Data Pull

This folder contains files for generating coarse data for the project and inference.

scan_twitter.ipynb - This notebook contains the code for pulling tweets from the Twitter API.
The coarse Twitter scan output is save in the output directory. The notebook uses arxiv_utils.py and tweet_utils.py.

Given a set of arXiv numbers (i.e. xxxx.xxxxx), it's possible to generate a dataset of abstracts 
ready for inference using Generate_custom_data_from_arxiv.ipynb

