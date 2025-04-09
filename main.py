"AWS Lambda code for returning Plant Disease Prediction for the given Image"

import pickle
from urllib.parse import unquote_plus

from tensorflow.keras.models import load_model # type: ignore

import boto3
from aws_lambda_powertools.utilities.data_classes import APIGatewayProxyEventV2
from aws_lambda_powertools.utilities.typing import LambdaContext

from constants import S3_BUCKET, BASE_USER_S3_DIR, PDD_MODEL_FILE, PDD_MODEL_PATH, PDD_BINARIZER_FILE, PDD_BINARIZER_PATH
from utils import getRandomUUID, isUUIDValid, response, getImage

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
    return response({ "obj_key": img_obj_key, "upload": res })

def get_inference(img_obj_key: str):
    "Returns the Result of the Prediction from the Model"

    data = getImage(img_obj_key)
    preds = model.predict(data)
    pred = label_binarizer.inverse_transform(preds)[0]
    return response(pred)

def handler(event: APIGatewayProxyEventV2, _context: LambdaContext):
    "Runs when Lambda is invoked using the Function URL"

    # Get Request Method
    method = event.get("http", {}).get("method") # type: ignore
    if method != "POST":
        return response("Invalid Method", 405)

    # Get Requested Route
    route = event.get("queryStringParameters", {}).get("route") # type: ignore
    if route is None:
        return response("Please provide a 'route' Query Parameter", 400)
    route = unquote_plus(route).strip()

    # Serve according to Route
    try:
        match route:
            case Routes.get("GET_UPLOAD_URL"):
                return get_upload_url()
            case Routes.get("GET_INFERENCE"):
                img_obj_key = event.get("body")
                if img_obj_key is None or not isUUIDValid(img_obj_key):
                    return response("Please provide a valid Image Key", 400)

                return get_inference(str(img_obj_key).strip())
            case _:
                return response(
                    f"'route' Parameter can either be `{Routes['GET_UPLOAD_URL']}` or `{Routes['GET_INFERENCE']}",
                    400
                )
    except Exception as e:
        print("[ERROR]:", e)
        return response("Internal Server Error", 500)
