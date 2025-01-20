from PIL import Image
from torchvision import transforms


preprocess = transforms.Compose([
    transforms.Resize(256), # Resizes the shorter side of the image to 256 pixels
    transforms.CenterCrop(224), # Crops a 224x224 region
    transforms.ToTensor(), # scales pixel values to the range [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]) # The final line is an alignment to the dataset ImageNet. The ResNet was trained there, so this matches the input distribution.


img = Image.open("test.png").convert("RGB")
input_tensor = preprocess(img).unsqueeze(0) # [1, 3, 224, 224]