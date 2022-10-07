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
