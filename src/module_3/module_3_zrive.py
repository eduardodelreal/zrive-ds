

import logging
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import boto3
import os

s3 = boto3.client(
    's3',
    #access key & password
)

bucket_name = 'zrive-ds-data'
file_key = 'groceries/box_builder_dataset/sampled_box_builder_df.csv'


local_directory = 'local_data'
if not os.path.exists(local_directory):
    os.makedirs(local_directory)

# Local file path where the file will be saved
local_file_path = os.path.join(local_directory, 'sampled_box_builder_df.csv')

# Download the file
s3.download_file(bucket_name, file_key, local_file_path)
print(f"Downloaded the file to {local_file_path}")

def main():
    pass
if __name__ == "__main__":
    main()