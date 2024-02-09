from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model.model_architecture import predict

app = FastAPI()

# Define the origins, methods, and headers that are allowed to access your API
# You can adjust these lists to match your requirements
origins = [
    "http://localhost:3000"
]

allowed_methods = ["*"]  # Adjust as needed
allowed_headers = ["*"]  # Include "application/json"

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=allowed_methods,
    allow_headers=allowed_headers,
)

@app.get("/")
async def root():
    return {"Hi there": '''Looks like you Found the Calorie Click API. Here are the steps to use it
    - Get the Base64 String of an Image (Try https://www.base64-image.de/)
    - Make a POST request to /predict
    - Get Classifying ;)
    '''}


@app.get("/sample_query")
async def sample_query(message: str = "Hello World"):
    """
    for output of this function, use the following command:
    localhost:8000/sample_query?message=message
    """
    return {"Message": message}


@app.post("/predict")
async def predict_class(base64_image: dict):
    """
    for output of this function, use the following command:
    localhost:8000/predict
    """
    try:
        base64_image = base64_image.get("base64_image")
        if not base64_image:
            raise HTTPException(
                status_code=400, detail="Missing 'base64_image' in the request body"
            )
        top_p, top_classes = predict(
            image_string=base64_image,
            topk=3,
            train_on_gpu=False,
            n_classes=14,
        )
        return {"top_classes": top_classes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
