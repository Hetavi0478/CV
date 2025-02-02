from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt  # Import matplotlib for later use (if needed)

app = FastAPI()

@app.post("/edge_detection")
async def detect_edges(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(image, (15, 15), 0)
        edge = cv2.Canny(blur, 100, 200)

        # Encode the edge image to JPEG format for streaming
        ret, encoded_image = cv2.imencode('.jpg', edge)
        if not ret:
            raise HTTPException(status_code=500, detail="Error encoding image")

        # Return the image as a stream
        return StreamingResponse(BytesIO(encoded_image.tobytes()), media_type="image/jpeg")

    except Exception as e:
        print(f"An error occurred: {e}")  # Print for debugging
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")



# Example of a route that returns the original image (optional)
@app.post("/original_image")
async def return_original_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        return StreamingResponse(BytesIO(contents), media_type=file.content_type)  # Directly stream the uploaded file

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


#If you want to display using matplotlib, you can create a separate endpoint and encode the image using base64
import base64
@app.post("/edge_detection_matplotlib")
async def detect_edges_matplotlib(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(image, (15, 15), 0)
        edge = cv2.Canny(blur, 100, 200)

        # Encode the edge image to base64 string
        _, encoded_image = cv2.imencode('.png', edge)  # Use PNG for lossless encoding
        encoded_string = base64.b64encode(encoded_image).decode('utf-8')


        return {"image_base64": encoded_string} # return the base64 encoded string

    except Exception as e:
        print(f"An error occurred: {e}")  # Print for debugging
        raise HTTPException
