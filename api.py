from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
import numpy as np

from transformers import SamModel, SamConfig, SamProcessor
import torch
import numpy as np
import cv2


def load_model_locally(path):
    """Loads retrained SAM model weights from local file
    and returns model ready to make a prediction"""
    # Load the model configuration
    model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
    # Create an instance of the model architecture with the loaded configuration
    model = SamModel(config=model_config)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model


def predict_mask(model, image:np.array) -> np.array:
    """Takes in a retrained SAM model and a google earth
    satellite image in numpy array format of any size and outputs a black and white image
    corresponding to rooftop masks. Output size is 256 by 256 pixels."""
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    if type(image) != np.array:
        image_array = np.array(image)
    else:
        image_array = image

    # Delete this splicing when running directly from google images
    #image_array = image_array[:,:,:-1]

    array_size = image_array.shape[0]
    # Higher grid sizes seem to confuse the model and decrease performance
    grid_size = 10
    # Generate grid points which will serve as prompt for SAM
    x = np.linspace(0, array_size-1, grid_size)
    y = np.linspace(0, array_size-1, grid_size)
    # Generate a grid of coordinates
    xv, yv = np.meshgrid(x, y)
    # Convert the numpy arrays to lists
    xv_list = xv.tolist()
    yv_list = yv.tolist()
    # Combine the x and y coordinates into a list of list of lists
    input_points = [[[int(x), int(y)] for x, y in zip(x_row, y_row)] for x_row, y_row in zip(xv_list, yv_list)]
    #We need to reshape our nxn grid to the expected shape of the input_points tensor
    # (batch_size, point_batch_size, num_points_per_image, 2),
    # where the last dimension of 2 represents the x and y coordinates of each point.
    #batch_size: The number of images you're processing at once.
    #point_batch_size: The number of point sets you have for each image.
    #num_points_per_image: The number of points in each set.
    input_points = torch.tensor(input_points).view(1, 1, grid_size*grid_size, 2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = processor(image_array, input_points=input_points, return_tensors="pt")
    # Move the input tensor to the GPU if it's not already there
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    # forward pass
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    # apply sigmoid
    mask_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
    # convert soft mask to hard mask
    mask_prob = mask_prob.cpu().numpy().squeeze()
    mask_prediction = (mask_prob > 0.5).astype(np.uint8)
    # Resize the mask from 256x256 to 572x572
    mask_prediction = cv2.resize(mask_prediction, dsize=(572, 572),
                                 interpolation=cv2.INTER_CUBIC)
    return mask_prediction

app = FastAPI()

# Define a Pydantic model to represent the input data
class InputData(BaseModel):
    image: np.array

    model_config = ConfigDict(arbitrary_types_allowed=True)

# Load the model
app.state.model = load_model_locally('weights.pth')
if app.state.model:
    print('Model loaded successfully!')


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define the route to accept POST requests with JSON data containing the Numpy array
@app.post("/predict")
def predict(data: InputData):
    # takes in a json dictionary via the pydantic basemodel object
    # the image key is then used to access the N-d list from the json
    # the list is then converted back to a numpy array and passed in to the prediction function
    try:
        print('Predict endpoint successfully reached!')
        print(f'Type of data being passed in: {type(data.image)}')
        print(f'Shape of input {np.array(data.image).shape}')
        outut_mask = predict_mask(model=app.state.model, image=data.image)
        return {"output_mask": outut_mask.tolist()} # Convert Numpy array to list for JSON serialization
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/')
def root():
    return {'api':'online'}