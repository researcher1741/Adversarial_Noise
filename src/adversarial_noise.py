"""
This file contains the implementation of the AdversarialNoiseGenerator class.
Key functionalities include:
- Generating adversarial noise.
- Predicting labels and class names for images.
- Saving adversarial images.
- Visualizing results.

The class supports ResNet18 with ImageNet-1K weights by default but can be extended for other models.
"""

# Libraries
import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
# Modules
from PIL import Image
from typing import Union
from pathlib import Path

class AdversarialNoiseGenerator:
    """
    This class generates adversarial examples by adding targeted noise to an input image.
    The noise is crafted to mislead the model into predicting a specified target class.
    """
    def __init__(self, epsilon: float = 0.01, print_labels: bool = False):
        """
        Initialize the generator with a specified noise magnitude (epsilon).
        """
        self.epsilon = epsilon
        self.print_labels = print_labels
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.eval()  # Set the model to evaluation mode by default
        self.preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()
        self.to_pil = ToPILImage()
        self.imagenet_classes = ResNet18_Weights.IMAGENET1K_V1.meta["categories"]

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

    def generate_adversarial_example_tensor(self, img_path: Union[Path, str], target_label: int) -> torch.Tensor:
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

    def generate_adversarial_example_image(self, img_path: Union[Path, str], target_label: int,
                                     output_path: Union[Path, str]) -> None:
        """
        Generate an adversarial example for a specified target label and save it as an image.
        Input:
            img_path (Union[Path, str]): Path to the input image.
            target_label (int): Target class index for misclassification.
            output_path (Union[Path, str]): Path to save the adversarial example.
        """
        adversarial_example = self.generate_adversarial_example_tensor(img_path, target_label)

        # Convert the adversarial tensor to an image and save it
        adversarial_img = self.to_pil(adversarial_example.squeeze(0))
        adversarial_img.save(output_path)

    def print_class_name(self, label_index: int) -> None:
        """
        Print the class name for a given label index based on ImageNet classes.
        Input:
            label_index (int): The label index for which to retrieve the class name.
        """
        if 0 <= label_index < len(self.imagenet_classes):
            class_name = self.imagenet_classes[label_index]
        else:
            class_name = "None"
        if self.print_labels:
            print(f"Label: {label_index}, Class Name: {class_name}")
        return class_name

    def predict_label(self, img_path: Union[Path, str]) -> int:
        """
        Predict the label for the input image.
        Input:
            img_path (Union[Path, str]): Path to the input image.
        Output:
            int: Predicted label index.
        """
        # Preprocess the image
        input_tensor = self.preprocess_img(img_path)

        # Get model predictions & Disable gradient computation for inference
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)

        # Get the index of the class with the highest score
        predicted_label = torch.argmax(output, dim=1).item()

        # Optionally print the class name
        class_name = self.imagenet_classes[predicted_label]
        if self.print_labels:
            print(f"Predicted Label: {predicted_label}, Class Name: {class_name}")

        return predicted_label, class_name
    @staticmethod
    def visualize_images(img_path: Union[Path, str], output_path: Union[Path, str],
                         label_in: int, class_name_in: str,
                         label_out: int, class_name_out: str,
                         target_label: int, class_name_target: str) -> None:
        """
        Visualize the input and adversarial images along with their labels and class names.

        Input:
            img_path (Union[Path, str]): Path to the input image.
            output_path (Union[Path, str]): Path to the adversarial image.
            label_in (int): Label of the input image.
            class_name_in (str): Class name of the input image.
            label_out (int): Label of the adversarial image.
            class_name_out (str): Class name of the adversarial image.
            target_label (int): Target label for adversarial misclassification.
            class_name_target (str): Class name of the target label.
        """
        input_img = Image.open(img_path)
        adversarial_img = Image.open(output_path)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(input_img)
        axs[0].axis("off")
        axs[0].set_title(f"Input Image\nLabel: {label_in} ({class_name_in})")

        axs[1].imshow(adversarial_img)
        axs[1].axis("off")
        axs[1].set_title(f"Adversarial Image\nLabel: {label_out} ({class_name_out})")

        plt.suptitle(f"Target Label: {target_label} ({class_name_target})", fontsize=14)
        plt.tight_layout()
        plt.show()


