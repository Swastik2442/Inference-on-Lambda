"Utilities"

import json
import uuid
from pathlib import Path

from tensorflow.keras.utils import load_img, img_to_array # type: ignore

import boto3

from constants import S3_BUCKET, USER_MEDIA_S3_DIR, USER_MEDIA_LOCAL_DIR, IMG_INPUT_SHAPE, IMG_SIZE

def response(data, error: str | None = None, status: int = 200):
    "Creates a simple JSON Response for the Lambda Function"
    return {
        "statusCode": status,
        "body": json.dumps({
            "status": "success" if status < 400 else "error",
            "data": data,
            "error": error,
        }),
        "isBase64Encoded": False,
        "headers": {"Content-Type": "application/json"},
    }

def getRandomUUID():
    "Generates a Random UUID"
    return uuid.uuid4().hex

def isUUIDValid(obj, version=4):
    "Checks whether the given UUID is Valid or not"
    try:
        uuid.UUID(obj, version=version)
        return True
    except ValueError:
        return False

um_dir_path = Path(USER_MEDIA_LOCAL_DIR)
def downloadImage(file_obj_key: str):
    "Downloads a particular Image from S3 if not present locally"
    assert isinstance(file_obj_key, str), "file_obj_key is required to be a String"
    assert isUUIDValid(file_obj_key), "file_obj_key must be a Valid UUID"

    um_dir_path.mkdir(exist_ok=True)
    file_path = Path(um_dir_path, file_obj_key)
    if not file_path.exists():
        boto3.client("s3").download_file(
            S3_BUCKET,
            USER_MEDIA_S3_DIR + file_obj_key,
            file_path.as_posix()
        )
    return file_path

def getImage(img_obj_key: str):
    "Returns the Image with the given `img_obj_key` as a Tensor"
    img_path = downloadImage(img_obj_key)
    img = load_img(img_path, target_size=IMG_SIZE)
    img_data = img_to_array(img)
    img_data = img_data.reshape(*IMG_INPUT_SHAPE) / 255.0
    return img_data
