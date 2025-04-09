"Utilities"

import json
import uuid
from pathlib import Path

from tensorflow.keras.utils import load_img, img_to_array # type: ignore

import boto3

from constants import WRITE_DIR, BASE_USER_S3_DIR, S3_BUCKET, IMG_INPUT_SHAPE, IMG_SIZE

def response(data, status=200):
    "Creates a simple JSON Response for the Lambda Function"
    return {
        "statusCode": status,
        "body": json.dumps({
            "status": "success" if status < 400 else "error",
            "data": data,
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
        uuidObj = uuid.UUID(obj, version=version)
    except ValueError:
        return False
    return str(uuidObj) == obj

def downloadImage(file_obj_key: str):
    "Downloads a particular Image from S3 if not present locally"
    assert isinstance(file_obj_key, str), "file_obj_key is required to be a String"
    assert isUUIDValid(file_obj_key), "file_obj_key must be a Valid UUID"

    file_path = Path(WRITE_DIR, BASE_USER_S3_DIR, file_obj_key)
    if not file_path.exists():
        boto3.client("s3").download_file(S3_BUCKET, BASE_USER_S3_DIR + file_obj_key, file_path)
    return file_path

def getImage(img_obj_key: str):
    "Returns the Image with the given `img_obj_key` as a Tensor"
    img_path = downloadImage(img_obj_key)
    img = img_to_array(load_img(img_path, IMG_SIZE))
    img = img.reshape(*IMG_INPUT_SHAPE) / 255.0
    return img
