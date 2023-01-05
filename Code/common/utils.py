from google.cloud import storage
import os

def format_from_cloud(path, bucket_name='paleo-ml'):
    '''
    Remove preceeding slashes and add a trailing slash to a path.
    It's important for the trailing slash to be there because it's used to
    determine if a path is a directory or a file. 
    '''
    assert path.startswith(f'gs://{bucket_name}')
    path = path[len(f'gs://{bucket_name}'):]
    while path.startswith('/'):
        path = path[1:]
    if not path.endswith('/'):
        path += '/'
    return path

def rename_cloud_file(original_name, new_name, bucket_name='paleo-ml'):
    # Names should start with gs://{bucket_name}
    original_name = format_from_cloud(original_name, bucket_name)
    new_name = format_from_cloud(new_name, bucket_name)

    storage_client = storage.Client()
    bucket = storage_client.bucket('paleo-ml')
    blob = bucket.blob(original_name)
    new_blob = bucket.rename_blob(blob, new_name)

def rename_cloud_dir(original_name, new_name, bucket_name='paleo-ml'):
    """
    Renames a directory in Google Cloud Storage. This is done by renaming all
    files in the directory and all subdirectories.
    """
    
    # Names should start with gs://{bucket_name}
    original_name = format_from_cloud(original_name, bucket_name)
    new_name = format_from_cloud(new_name, bucket_name)
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    objects = list(bucket.list_blobs(prefix=original_name))
    # objects contains all files in the directory and all subdirectories
    for obj in objects:
        if obj.name.endswith('/'):
            # Skip directories, as items in the directory will be renamed anyway
            continue
        new_blob_name = obj.name.replace(original_name, new_name)
        new_blob = bucket.rename_blob(obj, new_blob_name)

def download_dir(dir_name, bucket_name='paleo-ml'):
    """
    Download a directory from Google Cloud Storage.
    Args:
        dir_name (str): The name of the directory to download.
    Returns:
        str: The path to the downloaded directory.
    """
    print(dir_name)
    assert dir_name.startswith(f'gs://{bucket_name}')
    dir_name = dir_name[len(f'gs://{bucket_name}'):]

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=dir_name))
    for blob in blobs:
        if blob.name.endswith('/'):
            continue
        blob.download_to_filename(blob.name)
    return dir_name

def ls_cloud_dir(dir_name, bucket_name='paleo-ml'):
    """
    List the contents of a directory in Google Cloud Storage.
    Args:
        dir_name (str): The name of the directory to list.
    Returns:
        list: The names of the files in the directory.
    """
    dir_name = format_from_cloud(dir_name, bucket_name)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=dir_name))
    # The below is technically correct but doesn't work, so for now manually
    # filter out everything that's in a subdirectory.
    # blobs = list(bucket.list_blobs(delimiter='/', prefix=dir_name))
    
    ls_set = set()
    for blob in blobs:
        # skip if the blob is not a direct child of the directory
        if blob.name.count('/') > dir_name.count('/') + 1:
            continue
        ls_set.add(blob.name)
    
    return list(ls_set)
