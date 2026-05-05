from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import torch
from torchvision import transforms
from PIL import Image

class CustomModelWrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, inputs):
        # Assuming inputs is a tuple or list of tensors
        template, search = inputs
        return self.model(template, search)

# Load your pre-trained model
model = resnet50()
model = CustomModelWrapper(model)
model.eval()

# Define target layers
target_layers = [model.model.layer4[-1]]  # Adjust according to your model
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图片大小
    transforms.ToTensor()  # 转换为张量
])

# Prepare the input tensors
# template = torch.randn(1, 3, 224, 224)
# search = torch.randn(1, 3, 224, 224)
# 加载图片
image1 = Image.open("path_to_image1.jpg")
image2 = Image.open("path_to_image2.jpg")

# 预处理并转换为张量
template = preprocess(image1)
search = preprocess(image2)

input_tensors = (template, search)

# Initialize GradCAM
cam = GradCAM(model=model, target_layers=target_layers)

# Define the target class
targets = [ClassifierOutputTarget(281)]

# Compute the Grad-CAM
grayscale_cam = cam(input_tensor=input_tensors, targets=targets)

# Extract the first result
grayscale_cam = grayscale_cam[0, :]

# Visualization
rgb_img = search[0].permute(1, 2, 0).numpy()  # Assuming the visualization is based on the 'search' tensor
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
