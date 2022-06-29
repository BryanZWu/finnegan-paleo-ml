import os, sys, re, math
sys.path.append(os.getcwd() + '/..')
from common.constants import *


from data_processing import *
import pandas as pd
import numpy as np
np.random.seed(2022)
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import os, re, math
import keras
import shutil
from pathlib import Path
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds

print('TEST', dir_data)

def create_training_data(from_file_path, labels_df, data_dir, override, verbose=False):
    """
    Takes a single image file FROM_FILE_PATH and crops and resizes it
     to create a training data image. Write that to the corresponding directory.
    """
    
    ## Extract identifying information from the sample ##
    # from_file_path == drive/MyDrive/MV1012_SBB_images/Box_Core_images/MV1012-BC-8_identify/MV1012-BC-8_obj01142_plane000.jpg

    # MV1012-BC-8_obj01142.jpg
    image_file_name = os.path.basename(from_file_path).replace('_plane000', '')

    # MV1012-BC-8_obj01142
    image_name = Path(image_file_name).stem

    # MV1012-BC-8 01142
    sample_name, object_number = re.match(r'(.+)_obj(\d+)', image_name).groups()
    object_number = int(object_number)
    
    # TODO: exit if sample_name and object_number now found within labels_df
    if not (sample_name, object_number) in labels_df.index:
        if verbose: print(f'skipped {(sample_name, object_number)} becuase it is not in labels_df')
        return

    # can get label after confirming existence
    species_label = labels_df.loc[(sample_name, object_number), 'species']
    dataset_type = 'train'
    if labels_df.loc[(sample_name, object_number), 'test']:
        dataset_type = 'test'
    elif labels_df.loc[(sample_name, object_number), 'val']:
        dataset_type = 'val'


    ## Create empty dir or exit if it already exists and we don't want to override.
    img_location = os.path.join(data_dir, dataset_type, image_file_name)
    print(img_location)
    if os.path.exists(img_location):
        if not override: 
            if verbose: print(f'skipped {image_file_name}')
            return
        os.remove(img_location)

    img = Image.open(from_file_path)
    img_data = np.asarray(img)
    label_size = 160
    no_label_img_data = img_data[:-label_size]

    ## Find columns where a majority of the summed pixel intensities is 0
    zero_pixels = no_label_img_data.sum(axis=(2)) == 0
    col_filter = zero_pixels.sum(axis=0) < zero_pixels.shape[0]/5 #Cols where less than 1/5 pixels are zeros
    cropped_image_data = no_label_img_data[:, col_filter, :]
    cropped_image = Image.fromarray(cropped_image_data)
    ## Convert to image again and resize
    size = 416 # input size, width and height of image 
    resized_image = cropped_image.resize((size, size))

    ## Store image in output directory
    # print(os.path.join(data_point_dir, f'{image_name}.jpg'))
    resized_image.save(img_location)
    if verbose: print(f'created ${image_name} in {species_label} as {dataset_type}')


def process_sample_dir(sample_dir, sample_name, labels_df, data_dir=dir_dev_data, override=False, verbose=True):
    '''
    Takes a directory full of images and loops over them 
    to process the images.
    '''
    for file_name in os.listdir(sample_dir):
        file_path = os.path.join(sample_dir, file_name)
        file_ext = os.path.splitext(file_name)[-1].lower()
        if file_ext == '.csv':
            pass
        elif file_ext == '.jpg':
            #TODO: possible make the name the classification? 
            create_training_data(file_path, labels_df, data_dir=data_dir,override=override, verbose=verbose) #TODO set training data dir instead of using default debug

def purge(train_dir):
    print(f'purging all images in {train_dir}. Type "YES" to continue.')
    if input() != "YES":
        return
    for class_dir in os.listdir(train_dir):
        print(f'starting {class_dir}')
        total_deleted_in_class = 0
        class_path = os.path.join(train_dir, class_dir)
        for file in os.listdir(class_path):
            total_deleted_in_class += 1
            file_path = os.path.join(class_path, file)
            try:
                os.remove(file_path)
            except:
                shutil.rmtree(file_path)
        print(f'finished {class_dir} and removed {total_deleted_in_class}')
