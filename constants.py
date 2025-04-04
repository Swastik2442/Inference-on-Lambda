"Stores Constants required for Lambda to run"

from tensorflow.keras.backend import image_data_format # type: ignore

S3_BUCKET = "plantdiseasedetection"
WRITE_DIR = "/tmp/"
BASE_USER_S3_DIR = "user-media/"

PDD_MODEL_FILE = "cnn_model.keras"
PDD_BINARIZER_FILE = "label_transform.pkl"
PDD_MODEL_PATH = WRITE_DIR + PDD_MODEL_FILE
PDD_BINARIZER_PATH = WRITE_DIR + PDD_BINARIZER_FILE

IMG_SIZE = (256, 256)
IMG_INPUT_SHAPE = (1, *IMG_SIZE, 3)
if image_data_format() == "channels_first":
    IMG_INPUT_SHAPE = (1, 3, *IMG_SIZE)
