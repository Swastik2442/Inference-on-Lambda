"AWS Lambda code for returning Plant Disease Prediction for the given Image"

import pickle
from urllib.parse import unquote_plus
from typing import Dict, Any

from tensorflow.keras.models import load_model # type: ignore

import boto3
from aws_lambda_powertools import Logger
from aws_lambda_powertools.utilities.data_classes import LambdaFunctionUrlEvent
from aws_lambda_powertools.utilities.typing import LambdaContext

from constants import S3_BUCKET, BASE_USER_S3_DIR, PDD_MODEL_FILE, PDD_MODEL_PATH, PDD_BINARIZER_FILE, PDD_BINARIZER_PATH
from utils import getRandomUUID, isUUIDValid, response, getImage

Routes = {
    "GET_UPLOAD_URL": "upload",
    "GET_INFERENCE": "getinf"
}

logger = Logger("pdd-lambda", log_uncaught_exceptions=True)

logger.info("Downloading Model")
s3 = boto3.client("s3")
s3.download_file(S3_BUCKET, PDD_MODEL_FILE, PDD_MODEL_PATH)
s3.download_file(S3_BUCKET, PDD_BINARIZER_FILE, PDD_BINARIZER_PATH)

logger.info("Loading Model")
model = load_model(PDD_MODEL_PATH)
with open(PDD_BINARIZER_PATH, "rb") as file:
    label_binarizer = pickle.load(file)

def get_upload_url():
    "Returns an Image Key and Pre-Signed URL for uploading to S3"

    img_obj_key = getRandomUUID()
    res = s3.generate_presigned_url(
        ClientMethod='put_object',
        Params={
            'Bucket': S3_BUCKET,
            'Key': BASE_USER_S3_DIR + img_obj_key,
        },
        ExpiresIn=300 # 5 Mins
    )
    return response({ "obj_key": img_obj_key, "upload_url": res })

def get_inference(img_obj_key: str):
    "Returns the Result of the Prediction from the Model"

    data = getImage(img_obj_key)
    preds = model.predict(data)
    pred = label_binarizer.inverse_transform(preds)[0]
    return response(pred)

@logger.inject_lambda_context(log_event=True)
def handler(ev: Dict[str, Any], _ctx: LambdaContext):
    "Runs when Lambda is invoked using the Function URL"

    event = LambdaFunctionUrlEvent(ev)

    # Get Requested Route
    route = event.query_string_parameters.get("route") # type: ignore
    if route is None:
        return response(None, "Please provide a 'route' Query Parameter", 400)

    route = unquote_plus(route).strip().lower()
    if route not in Routes.values():
        return response(
            None, f"'route' Parameter can either one of {tuple(Routes.values())}", 400
        )

    # Serve according to Route
    try:
        if route == Routes["GET_UPLOAD_URL"]:
            if event.http_method != "GET":
                return response(None, "Invalid Method", 405)
            return get_upload_url()

        if route == Routes["GET_INFERENCE"]:
            if event.http_method != "POST":
                return response(None, "Invalid Method", 405)

            obj_key = event.json_body.get("obj_key")
            if obj_key is None or not isUUIDValid(obj_key):
                return response(None, "Please provide a valid Image Key", 400)
            return get_inference(str(obj_key).strip())
    except Exception as e:
        logger.error(e)
        return response(None, "Internal Server Error", 500)
