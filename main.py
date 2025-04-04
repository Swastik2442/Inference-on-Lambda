"AWS Lambda code for returning Plant Disease Prediction for the given Image"

import pickle
from urllib.parse import unquote_plus

from tensorflow.keras.models import load_model # type: ignore

import boto3
from aws_lambda_powertools.utilities.data_classes import APIGatewayProxyEventV2
from aws_lambda_powertools.utilities.typing import LambdaContext

from constants import *
from utils import *

Routes = {
    "GET_UPLOAD_URL": "upload",
    "GET_INFERENCE": "getinf"
}

# Download Model
s3 = boto3.client("s3")
s3.download_file(S3_BUCKET, PDD_MODEL_FILE, PDD_MODEL_PATH)
s3.download_file(S3_BUCKET, PDD_BINARIZER_FILE, PDD_BINARIZER_PATH)

# Create Label Binarizer
with open(PDD_BINARIZER_PATH, "rb") as file:
    label_binarizer = pickle.load(file)

model = load_model(PDD_MODEL_PATH)

def get_upload_url():
    img_obj_key = getRandomUUID()
    res = s3.generate_presigned_url(
        ClientMethod='put_object',
        Params={
            'Bucket': S3_BUCKET,
            'Key': BASE_USER_S3_DIR + img_obj_key,
        },
        ExpiresIn=300 # 5 Mins
    )
    return response({ "obj_key": img_obj_key, "upload": res })

def get_inference(img_obj_key: str):
    data = getImage(img_obj_key)
    preds = model.predict(data)
    pred = label_binarizer.inverse_transform(preds)[0]
    return response(pred)

def handler(event: APIGatewayProxyEventV2, _context: LambdaContext):
    "Runs when Lambda is invoked using the Function URL"

    try:
        method = event.get("http", {}).get("method")
        if method != "POST":
            return response("Invalid Method", 405)

        route = event.get("queryStringParameters", {}).get("route")
        if route is None:
            return response("Please provide a 'route' Query Parameter", 400)

        route = unquote_plus(route).strip()
        if route == Routes["GET_UPLOAD_URL"]:
            return get_upload_url()
        elif route == Routes["GET_INFERENCE"]:
            img_obj_key = event.get("body").strip()
            return get_inference(img_obj_key)
        else:
            return response(f"'route' Parameter can either be `{Routes["GET_UPLOAD_URL"]}` or `{Routes["GET_INFERENCE"]}`", 400)
    except Exception as e:
        print("[ERROR]:", e)
        return response("Internal Server Error", 500)
