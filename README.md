# Inference on Lambda

**A lightweight template for deploying TensorFlow image classification models on AWS Lambda using Docker (ECR) and S3 for serverless inference.**

---

## 🚀 Architecture Overview

1. A TensorFlow model is trained and saved locally.
2. The model and label binarizer are uploaded to an S3 bucket.
3. A Dockerized Lambda function loads the model and serves predictions.
4. Clients use the Lambda URL to:
   - Get a signed S3 URL and image key for upload.
   - Trigger inference on the uploaded image via a REST API.

---

## ⚙️ Setup Instructions

### 1. 🧪 Train and Export Model

Train your image classification model using TensorFlow, then save it:

```python
model.save("model")
```

Create and save the label binarizer:

```python
import pickle

with open("label_binarizer.pkl", "wb") as f:
    pickle.dump(label_binarizer, f)
```

### 2. ☁️ Upload to S3

Upload the saved model directory and the `label_binarizer.pkl` file to your S3 bucket.

### 3. ⚙️ Configure Constants

Edit [`constants.py`](constants.py) and set:

* `BUCKET_NAME`: your S3 bucket name
* `MODEL_PATH`: path to your model on S3
* `LABEL_BINARIZER_PATH`: path to the `.pkl` file
* Other settings as required

## 🔌 API Endpoints

Base URL: `<LAMBDA_FUNCTION_URL>`

### 📤 `GET /?route=upload`

* **Returns:**
  * `upload_url`: Pre-signed S3 URL for image upload
  * `obj_key`: Unique image key
* **Purpose:** Enables direct upload of images to S3 from the client.

### 🧠 `POST /?route=get_inf`

* **Request Body:**
  ```json
  {
    "obj_key": "your_uploaded_image_key.jpg"
  }
  ```
* **Returns:**
  * The predicted class label with the highest confidence score.

---

## 📸 Example Usage

```bash
# Step 1: Get a pre-signed URL and image key
curl <LAMBDA_FUNCTION_URL>/?route=upload

# Step 2: Upload your image to S3
curl -X PUT -T image.jpg "<UPLOAD_URL_FROM_STEP_1>"

# Step 3: Trigger inference
curl -X POST <LAMBDA_FUNCTION_URL>/?route=get_inf \
     -H "Content-Type: application/json" \
     -d '{"obj_key": "<IMAGE_KEY_FROM_STEP_1>"}'
```

---

## 📜 License

This project template is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute for personal or commercial use.
