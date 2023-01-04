from google.cloud import storage

def rename_cloud_file(original_name, new_name, bucket_name='paleo-ml'):
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The ID of the GCS object to rename
    # blob_name = "your-object-name"
    # The new ID of the GCS object
    # new_name = "new-object-name"
    
    # Names should start with gs://{bucket_name}
    assert original_name.startswith(f'gs://{bucket_name}')
    assert new_name.startswith(f'gs://{bucket_name}')
    original_name = original_name[len(f'gs://{bucket_name}'):]
    new_name = new_name[len(f'gs://{bucket_name}'):]
    if original_name.startswith('/'):
        original_name = original_name[1:]
    if new_name.startswith('/'):
        new_name = new_name[1:]

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
    assert original_name.startswith(f'gs://{bucket_name}')
    assert new_name.startswith(f'gs://{bucket_name}')
    original_name = original_name[len(f'gs://{bucket_name}'):]
    new_name = new_name[len(f'gs://{bucket_name}'):]
    if original_name.startswith('/'):
        original_name = original_name[1:]
    if new_name.startswith('/'):
        new_name = new_name[1:]
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    objects = list(bucket.list_blobs(prefix=original_name))
    # objects contains all files in the directory and all subdirectories
    for obj in objects:
        if obj.name.endswith('/'):
            # Skip directories
            continue
        new_blob_name = obj.name.replace(original_name, new_name)
        new_blob = bucket.rename_blob(obj, new_blob_name)

def download_dir(dir_name):
    """
    Download a directory from Google Cloud Storage.
    Args:
        dir_name (str): The name of the directory to download.
    Returns:
        str: The path to the downloaded directory.
    """
    # The ID of your GCS bucket
    bucket_name = "paleo-ml"
    # The ID of the GCS object to download
    # blob_name = "your-object-name"
    # The path to which the file should be downloaded
    # download_to_filename = "local/path/to/file"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=dir_name)
    for blob in blobs:
        if blob.name.endswith('/'):
            continue
        blob.download_to_filename(blob.name)
    return dir_name