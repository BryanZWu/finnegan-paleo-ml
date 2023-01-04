from google.cloud import storage

def rename_cloud_file(original_name, new_name):
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The ID of the GCS object to rename
    # blob_name = "your-object-name"
    # The new ID of the GCS object
    # new_name = "new-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket('paleo-ml')
    blob = bucket.blob(original_name)
    new_blob = bucket.rename_blob(blob, new_name)

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