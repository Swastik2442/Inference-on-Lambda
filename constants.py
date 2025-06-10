"Stores Constants required for Lambda to run"

from tensorflow.keras.backend import image_data_format # type: ignore

S3_BUCKET = "your-s3-bucket-name"                    # S3 bucket to store user media files
WRITE_DIR = "/tmp/"                                  # Directory to write files in Lambda
USER_MEDIA_S3_DIR = "user-media/"                    # S3 directory to store user media files
USER_MEDIA_LOCAL_DIR = WRITE_DIR + USER_MEDIA_S3_DIR # Local directory to store user media files

MODEL_FILE = "my_model.keras"                        # Model file to be used for inference
BINARIZER_FILE = "label_transform.pkl"               # Binarizer file to be used for inference
MODEL_PATH = WRITE_DIR + MODEL_FILE                  # Local path to store the model file
BINARIZER_PATH = WRITE_DIR + BINARIZER_FILE          # Local path to store the binarizer file

IMG_SIZE = (256, 256)                                # Size of the input image (height, width)
IMG_INPUT_SHAPE = (1, *IMG_SIZE, 3)                  # Input shape for the model (batch_size, height, width, channels)
if image_data_format() == "channels_first":
    IMG_INPUT_SHAPE = (1, 3, *IMG_SIZE)
