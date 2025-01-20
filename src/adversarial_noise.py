import torch
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from typing import Union
from pathlib import Path

class AdversarialNoiseGenerator:
    """
    This class generates adversarial examples by adding targeted noise to an input image.
    The noise is crafted to mislead the model into predicting a specified target class.
    """
    def __init__(self, epsilon: float = 0.01):
        """
        Initialize the generator with a specified noise magnitude (epsilon).
        """
        self.epsilon = epsilon
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.eval()  # Set the model to evaluation mode by default
        self.preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()

    def preprocess_img(self, img_path: Union[Path, str]) -> torch.Tensor:
        """
        Preprocess the input image to prepare it for the model.
        Input:
            img_path (Path or str): Path to the input image.
        Output:
            input_tensor (torch.tensor):Preprocessed image tensor.
        """
        img = Image.open(img_path).convert("RGB")  # Ensure the image is in RGB format
        input_tensor = self.preprocess(img).unsqueeze(0)  # Add batch dimension
        return input_tensor

    def generate_adversarial_noise(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Generate adversarial noise for the input tensor.
        Input:
            input_tensor (torch.tensor): Tensor with gradients enabled.
        Output:
            adversarial_example (torch.tensor): Adversarial example tensor.
        """
        adversarial_noise = self.epsilon * input_tensor.grad.sign()
        adversarial_example = input_tensor + adversarial_noise
        adversarial_example = torch.clamp(adversarial_example, 0, 1)  # Keep values in valid range
        return adversarial_example

    def generate_adversarial_example(self, img_path: Union[Path, str], target_label: int) -> torch.Tensor:
        """
        Generate an adversarial example for a specified target label.
        Input:
            img_path (Union[Path, str]): Path to the input image.
            target_label (int): Target class index for misclassification.
        Output:
            adversarial_example (torch.tensor): Adversarial image tensor.
        """
        input_tensor = self.preprocess_img(img_path)
        input_tensor.requires_grad = True

        # Define the target label as a tensor
        target = torch.tensor([target_label])

        # Forward pass and loss calculation
        output = self.model(input_tensor)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        loss.backward()  # Compute gradients

        # Generate adversarial example
        adversarial_example = self.generate_adversarial_noise(input_tensor)

        return adversarial_example
