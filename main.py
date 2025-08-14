from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io

# 1. DEFINE THE MODEL ARCHITECTURE (must be the same as in your notebook)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 2. LOAD THE TRAINED MODEL WEIGHTS
net = Net()
PATH = './cifar_net.pth'
net.load_state_dict(torch.load(PATH))
net.eval()  # Set the model to evaluation mode

# 3. DEFINE IMAGE TRANSFORMS AND CLASS NAMES
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 4. CREATE THE FASTAPI APP
app = FastAPI(title="CIFAR-10 Image Classifier API")

@app.get("/")
def read_root():
    return {"message": "Welcome to the CIFAR-10 Image Classifier API. Go to /docs for usage."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image contents from the uploaded file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Process the image and make a prediction
    with torch.no_grad():
        input_tensor = transform(image).unsqueeze(0)
        outputs = net(input_tensor)
        _, predicted_idx = torch.max(outputs.data, 1)
        predicted_class = classes[predicted_idx.item()]

    return {"predicted_class": predicted_class}