import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.constants import *
from common.imports import *
# import forams

def create_labels_df(csv_labels): 
    """
    Reads master CSV and split into train-val-test
    """
    labels_df = pd.read_csv(csv_labels)
    labels_df = labels_df.set_index(['sample_name', 'object_num'])
    labels_df.sort_index()
    # ACCESS via: labels_df.loc[('MV1012-BC-2', 1)]
    labels_df = labels_df.dropna(how='any')

    # Randomly decide if it will be in the train, val or test section.
    # 0.8 training, 0.2 test
    # 0.8 * 0.8 = 0.64 train, 0.16 val. 
    # TODO: decide if we want to keep it at these particular values
    sample_ind = np.random.random_sample(labels_df.shape[0])
    labels_df['test'] = sample_ind > 0.8
    labels_df['val'] = (sample_ind <= 0.8) & (sample_ind > 0.64)
    return labels_df

def create_training_data(from_file_path, labels_df, dir_processed_data, override, verbose=False):
    """
    Takes a single image file FROM_FILE_PATH and crops and resizes it
    to create a training data image. Write that to the corresponding directory.
    """
    # example from_file_path == drive/MyDrive/MV1012_SB_images/Box_Core_images/MV1012-BC-8_identify/MV1012-BC-8_obj01142_plane000.jpg
    
    ## Extract identifying information from the sample ##

    # MV1012-BC-8_obj01142.jpg
    image_file_name = os.path.basename(from_file_path).replace('_plane000', '')

    # MV1012-BC-8_obj01142
    image_name = Path(image_file_name).stem

    # MV1012-BC-8 01142
    sample_name, object_number = re.match(r'(.+)_obj(\d+)', image_name).groups()
    object_number = int(object_number)
    
    # TODO: exit if sample_name and object_number not found within labels_df
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


    # Create empty dir or exit if it already exists and we don't want to override.
    img_location = os.path.join(dir_processed_data, dataset_type, image_file_name)
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
    size = 224 # input size, width and height of image 
    resized_image = cropped_image.resize((size, size))

    ## Store image in output directory
    # print(os.path.join(data_point_dir, f'{image_name}.jpg'))
    resized_image.save(img_location)
    if verbose: print(f'created ${image_name} in {species_label} as {dataset_type}')


def process_sample_dir(sample_dir, sample_name, labels_df, dir_processed_data, override=False, verbose=True):
    '''
    Takes a directory full of images and loops over them to process the images.

    '''
    for file_name in os.listdir(sample_dir):
        file_path = os.path.join(sample_dir, file_name)
        file_ext = os.path.splitext(file_name)[-1].lower()
        if file_ext == '.csv':
            pass
        elif file_ext == '.jpg':
            #TODO: possible make the name the classification? 
            create_training_data(file_path, labels_df, dir_processed_data=dir_processed_data,override=override, verbose=verbose)

def purge(dir_split_to_purge):
    """
    Note: when complete this should only purge the google drive filesys copies but not the 
    google cloud ones.
    """
    print(f'purging all images in {dir_split_to_purge}. Type "YES" to continue.')
    if input() != "YES":
        return

    return print("This function is not up to date with custom tensorflow dataset update. Please implement this function.")
    for class_dir in os.listdir(dir_split_to_purge):
        print(f'starting {class_dir}')
        total_deleted_in_class = 0
        class_path = os.path.join(dir_split_to_purge, class_dir)
        for file in os.listdir(class_path):
            total_deleted_in_class += 1
            file_path = os.path.join(class_path, file)
            try:
                os.remove(file_path)
            except:
                shutil.rmtree(file_path)
        print(f'finished {class_dir} and removed {total_deleted_in_class}')

def run_processing(dir_processed_data, dir_raw_training_images, labels_df):
    dir_exists = os.path.isdir(dir_processed_data)
    if not dir_exists:
        print(f'unable to find the output for processed training data: {dir_processed_data}. Type "YES" to create it and continue.')
        if input() != "YES":
            print('processing aborted')
            return
        os.makedirs(f'{dir_processed_data}/train')
        os.makedirs(f'{dir_processed_data}/val')
        os.makedirs(f'{dir_processed_data}/test')


    for sample_dir in os.listdir(dir_raw_training_images):
        sample_dir_path = os.path.join(dir_raw_training_images, sample_dir)
        sample_name = re.match(r'(.+)_identify', sample_dir).groups()[0]
        process_sample_dir(sample_dir_path, sample_name, labels_df, dir_processed_data, override=False)


def create_cloud_dataset(dir_local_processed_data, dir_cloud_data, dir_dataset_specs, batch_size):
    # TODO: support for dataset data that's not in the same dir as specs
    cur_cwd = os.getcwd()
    os.chdir(dir_dataset_specs)
    import forams
    os.chdir(cur_cwd)
    training_set = tfds.load('forams', split='train')
    validation_set = tfds.load('forams', split='val')
    testing_set = tfds.load('forams', split='test')
    print(training_set)
    # os.chdir(cur_cwd)
    tf.data.experimental.save(training_set, f'{dir_cloud_data}/training{batch_size}')
    tf.data.experimental.save(validation_set, f'{dir_cloud_data}/validation{batch_size}')
    tf.data.experimental.save(testing_set, f'{dir_cloud_data}/testing{batch_size}')

    # Above code deprecated below not supported in colab default tf version
    # tf.data.Dataset.save(training_set, f'{dir_cloud_data}/training{batch_size}')
    # tf.data.Dataset.save(validation_set, f'{dir_cloud_data}/validation{batch_size}')
    # tf.data.Dataset.save(testing_set, f'{dir_cloud_data}/testing{batch_size}')

def load_cloud_dataset(dir_cloud_data, batch_size):
    '''
    Loads the dataset from the cloud storage bucket. Returns the training, validation, and testing sets.
    '''
    training_set = tf.data.experimental.load(f'{dir_cloud_data}/training{batch_size}').cache().prefetch(tf.data.experimental.AUTOTUNE)
    validation_set = tf.data.experimental.load(f'{dir_cloud_data}/validation{batch_size}').cache().prefetch(tf.data.experimental.AUTOTUNE)
    testing_set = tf.data.experimental.load(f'{dir_cloud_data}/testing{batch_size}').cache().prefetch(tf.data.experimental.AUTOTUNE)
    # testing_set = None

    # Above code deprecated below not supported in colab default tf version
    # training_set = tf.data.Dataset.load(f'{dir_cloud_data}/training').cache().prefetch(tf.data.experimental.AUTOTUNE)
    # validation_set = tf.data.Dataset.load(f'{dir_cloud_data}/validation').cache().prefetch(tf.data.experimental.AUTOTUNE)
    # testing_set = tf.data.Dataset.load(f'{dir_cloud_data}/testing').cache().prefetch(tf.data.experimental.AUTOTUNE)
    return training_set, validation_set, testing_set
