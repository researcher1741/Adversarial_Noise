from src.adversarial_noise import AdversarialNoiseGenerator
from pathlib import Path


if __name__ == "__main__":
    # Example usage
    img_path = "Data/panda.png"  # Path to the input image
    target_label = 207  # Specify the target class index
    input_path1 = Path(img_path)
    output_path = str(input_path1.parent / f"adversarial_output_{input_path1.stem}.jpg")

    # Instance generator
    generator = AdversarialNoiseGenerator(epsilon=0.2)

    # Check the label of the input image and the target label
    label_in, class_name_in = generator.predict_label(img_path)
    print(f"The input image label is: {label_in}, {class_name_in}")
    class_name_target = generator.print_class_name(target_label)
    print(f"The Target label is: {target_label}, {class_name_target}")

    # Generation
    generator.generate_adversarial_example_image(img_path, target_label, output_path)

    # Result image
    label_out, class_name_out = generator.predict_label(output_path)
    print(f"The Output image label is: {label_out}, {class_name_out}")
    if label_in != label_out:
        print(f"We got a misclassification!")
        if target_label == label_out:
            print(f"We got a misclassification of the input with the target label at the output!")
        else:
            print(f"At least we got a misclassification, but not to the correct output label")

    print(f"Adversarial image saved to {output_path}")