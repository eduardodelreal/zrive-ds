

import logging
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import boto3
import os

'''
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
'''
#data=pd.read_csv("feature_frames.csv")
#data = pd.read_csv("feature_frames.csv", delimiter=",")
#data = pd.read_csv("feature_frames.csv")
data = pd.read_csv("feature_frames.csv")
#print(data)



# Filter orders with more than five products ordered
#filtered_data = data[data['user_order_seq'] > 5]

# Save the filtered dataset to a new CSV file
#filtered_data.to_csv("filtered_feature_frames.csv", index=False)
def main():
    print(data)
    #original_shape = data.shape

# Obtener el tamaño del dataset filtrado
    #filtered_shape = filtered_data.shape

# Imprimir los tamaños para comparar
    #print("Tamaño del dataset original:", original_shape)
    #print("Tamaño del dataset filtrado:", filtered_shape)
    #pass
if __name__ == "__main__":
    main()
