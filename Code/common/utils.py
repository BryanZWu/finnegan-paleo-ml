from google.cloud import storage
import os
import json

def format_dirs(path):
    '''
    Remove preceeding slashes and add a trailing slash to a path.
    It's important for the trailing slash to be there because it's used to
    determine if a path is a directory or a file in the Google Cloud Storage
    '''
    while path.startswith('/'):
        path = path[1:]
    if not path.endswith('/'):
        path += '/'
    return path

def rename_cloud_file(original_name, new_name, bucket_name='paleo-ml'):
    # Names should start with gs://{bucket_name}
    original_name = format_dirs(original_name)
    new_name = format_dirs(new_name)

    storage_client = storage.Client()
    bucket = storage_client.bucket('paleo-ml')
    blob = bucket.blob(original_name)
    new_blob = bucket.rename_blob(blob, new_name)

def rename_cloud_dir(original_name, new_name, bucket_name='paleo-ml'):
    """
    Renames a directory in Google Cloud Storage. This is done by renaming all
    files in the directory and all subdirectories.
    """

    original_name = format_dirs(original_name)
    new_name = format_dirs(new_name)
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    objects = list(bucket.list_blobs(prefix=original_name))
    # sort by length of the name so that the longest names are renamed first
    objects.sort(key=lambda x: len(x.name), reverse=True)
    # objects contains all files in the directory and all subdirectories
    for obj in objects:
        if obj.name.endswith('/'):
            # Skip directories, as items in the directory will be renamed anyway
            continue
        new_blob_name = obj.name.replace(original_name, new_name)
        new_blob = bucket.rename_blob(obj, new_blob_name)
    
    # Delete the directories and subdirectories by sorting by length of the 
    # directory name and deleting the longest names first
    remaining_objects = list(bucket.list_blobs(prefix=original_name))
    remaining_objects.sort(key=lambda x: len(x.name), reverse=True)
    for obj in remaining_objects:
        obj.delete()

def download_dir(dir_name, bucket_name='paleo-ml', save_dir='/temp/'):
    """
    Download a directory from Google Cloud Storage and return the path to the
    downloaded directory.

    Args:
        dir_name (str): The name of the directory to download.
        bucket_name (str): The name of the bucket to download from.
        save_dir (str): The directory to save the downloaded files to.
    Returns:
        str: The path to the downloaded directory.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=dir_name))
    for blob in blobs:
        if blob.name.endswith('/'):
            continue
        filename = blob.name.split('/')[-1]
        save_path = os.path.join(save_dir, dir_name, filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        blob.download_to_filename(save_path)
    return os.path.join(save_dir, dir_name)

def ls_cloud_dir(dir_name, bucket_name='paleo-ml'):
    """
    List the contents of a directory in Google Cloud Storage.
    Args:
        dir_name (str): The name of the directory to list.
    Returns:
        list: The names of the files in the directory. Each one is a complete
            path, including the directory name.
    """
    dir_name = format_dirs(dir_name)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=dir_name))
    # The below is technically correct but doesn't work, so for now manually
    # filter out everything that's in a subdirectory.
    # blobs = list(bucket.list_blobs(delimiter='/', prefix=dir_name))

    num_slashes = dir_name.count('/')
    
    ls_set = set()
    for blob in blobs:
        segments = blob.name.split('/')
        # Take the segments after the directory name, or skip if there's 
        # nothing after the directory name

        # example: if dir_name is 'a/b/c/' and blob.name is 'a/b/c/d/e/f',
        # then segments is ['a', 'b', 'c', 'd', 'e', 'f'] and we want to take
        # 'd'. If blob.name is 'a/b/c', then we want to skip it.

        entry = segments[num_slashes]
        if entry == '':
            continue

        # If the entry is a directory, add a trailing slash
        if len(segments) > num_slashes + 1 or blob.name.endswith('/'):
            entry += '/'

        ls_set.add(entry)
    return list(ls_set)

def parse_gcs_path(dir):
    '''
    Returns the bucket name and the path within the bucket.
    '''
    assert dir.startswith('gs://')
    dir = dir[len('gs://'):]
    bucket_name, path = dir.split('/', 1)
    return bucket_name, path

def get_gcs_path(local_path, bucket_name='paleo-ml'):
    '''
    Returns the Google Cloud Storage path for a local path.
    '''
    return f'gs://{bucket_name}/{local_path}'

def json_dump_gcs(obj, path, bucket_name='paleo-ml'):
    '''
    Dumps a JSON object to a file in Google Cloud Storage.
    '''
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(path)
    blob.upload_from_string(json.dumps(obj))
    
