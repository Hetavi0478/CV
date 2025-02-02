import cv2
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os

app = FastAPI()

@app.post("/upload_image/")
async def upload_image(file: UploadFile = File(...)):
    # Save the uploaded image temporarily
    temp_input_path = "temp_input.jpg"
    with open(temp_input_path, "wb") as f:
        f.write(await file.read())

    # Read the uploaded image
    img = cv2.imread(temp_input_path)

    # Convert the image to grayscale (necessary for edge detection)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Convert BGR to RGB for proper display in matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Plotting the images using matplotlib and saving it as a temporary file
    output_path = "output_image.jpg"
    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')

    # Edge Detection Image
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Detection')
    plt.axis('off')

    # Save the figure as a temporary image file
    plt.savefig(output_path)
    plt.close()

    # Remove the temporary input image
    os.remove(temp_input_path)

    # Return the processed image as a FileResponse
    return FileResponse(output_path, media_type='image/jpeg', filename='processed_image.jpg')

