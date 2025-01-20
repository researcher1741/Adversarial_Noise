# Adversarial_Noise

## Introduction
This library provides a simple interface for generating adversarial examples by introducing imperceptible noise to input images. 
These adversarial images are designed to trick a pre-trained image classification model into misclassifying the image 
as a user-specified target class, regardless of the original content.

## Features
- Generate adversarial noise for any user-provided image.
- Specify the desired target class for misclassification.
- Use pre-trained models from the `torchvision` library.
- Ensure generated adversarial images remain visually similar to the original.

## Requirements
- Python 3.8 or later
- PyTorch
- torchvision
- PIL (Python Imaging Library)

## Installation
Clone this repository and install the required dependencies:
```bash
# Clone the repository
git clone https://github.com/researcher1741/Adversarial_Noise
cd Adversarial_Noise

# Install dependencies
pip install -r requirements.txt
```

## Usage
Here is how you can use the library to generate adversarial examples:

### Example:
```python
from adversarial_noise import AdversarialNoiseGenerator

# Initialize the generator with a pre-trained model
generator = AdversarialNoiseGenerator(model_name='resnet18', epsilon=0.0045)

# Provide an input image and target class
adversarial_image = generator.generate("Data/input_image.jpg", target_class=207)

# Save or visualize the adversarial image
adversarial_image.save("Data/output_image.jpg")
```

## Inputs
1. **Input Image**: A file path to the image to be perturbed. It might be located in the folder Data.
2. **Target Class**: An integer specifying the target class the model should misclassify the image as.

## Outputs
- An adversarial image that the model classifies as the specified target class. 
- The image is saved in the specified output path.

## How It Works
1. **Load the Image**: The input image is preprocessed to match the model's input requirements.
2. **Compute Adversarial Noise**: Noise is calculated using the Fast Gradient Sign Method (FGSM) for the target case
3. **Generate Adversarial Image**: The noise is added to the image.
4. **Validate Misclassification**: The adversarial image is checked to ensure it is classified as the target class by the model.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
