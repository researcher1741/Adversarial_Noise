from src.adversarial_noise import AdversarialNoiseGenerator
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == "__main__":
    # Example usage
    img_path = "Data/panda.png"  # Path to the input image
    target_label = 368  # Specify the target class index
    input_path1 = Path(img_path)
    output_path = str(input_path1.parent / f"adversarial_output_{input_path1.stem}.jpg")

    # Instance generator
    generator = AdversarialNoiseGenerator(epsilon=0.09, print_labels=True)

    # Check the label of the input image
    label_in, class_name_in = generator.predict_label(img_path)
    print(f"The input image label is: {label_in}, {class_name_in}")

    # Check the target label
    class_name_target = generator.print_class_name(target_label)
    print(f"The Target label is: {target_label}, {class_name_target}")

    # Generate adversarial example
    generator.generate_adversarial_example_image(img_path, target_label, output_path)

    # Check the label of the output image
    label_out, class_name_out = generator.predict_label(output_path)
    print(f"The Output image label is: {label_out}, {class_name_out}")
    print(f"Adversarial image saved to {output_path}")
    generator.visualize_images(img_path, output_path, label_in, class_name_in, label_out,
                               class_name_out, target_label, class_name_target)
