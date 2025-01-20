from src.adversarial_noise import AdversarialNoiseGenerator


if __name__ == "__main__":
    # Example usage
    img_path = "Data/Input_image.png"  # Path to the input image
    target_label = 207  # Specify the target class index
    output_path = "Data/adversarial_output.jpg"  # Path to save the adversarial image

    generator = AdversarialNoiseGenerator(epsilon=0.01)
    generator.generate_adversarial_example_image(img_path, target_label, output_path)

    print(f"Adversarial image saved to {output_path}")

    generator = AdversarialNoiseGenerator(epsilon=0.01)

    # Print class name for a label
    generator.print_class_name(target_label)