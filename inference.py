import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from diffusion import GaussianDiffusion
from transModel import registUnetBlock

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = Image.open(image_path).convert("L")
    image = transform(image)
    return image.unsqueeze(0).to(device)

def tensor_to_image(tensor):
    """
    Convert a single-image tensor to a numpy array [H, W].
    """
    tensor = tensor.squeeze(0).detach().cpu().numpy()  # shape [1, H, W] or [H, W]
    if len(tensor.shape) == 3:
        tensor = tensor[0, ...]  # if shape is [1,H,W], take the first channel
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
    return (tensor * 255).astype(np.uint8)

def main():
    # Suppose you have saved your trans model as:
    # trans = torch.load("trans_20250210_paper04.pth", map_location=device)
    # trans.eval()
    trans_path = "trans_20250210_paper04.pth"  # replace with your path
    trans = torch.load(trans_path, map_location=device)
    trans.eval()

    # Suppose you have a saved average score:
    score_ave_path = "score_04.pt"
    score_ave = torch.load(score_ave_path, map_location=device)
    score_ave = score_ave.mean(dim=0, keepdim=True) * 0.9

    # We take an X-ray image for inference
    test_image_path = r"C:\Users\wz\gpu\OAI_m\paper_0.PNG"
    x_input = preprocess_image(test_image_path)

    # Create a diffusion helper (no need for losses in inference)
    gaussian_diffusion = GaussianDiffusion(timesteps=300)

    with torch.no_grad():
        # This calls p_sample_loop_validation_0_4 internally
        code_stack, defm_stack, flow_stack = gaussian_diffusion.sample_validation(
            trans, x_input, score_ave
        )

    # Plot results
    plt.figure(figsize=(15, 5))

    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(tensor_to_image(x_input), cmap="gray")
    plt.title("Original X-ray")
    plt.axis("off")

    # defm_stack is a stack with multiple images: [batch*(some gamma steps?), c, h, w]
    # For demonstration, let's pick index=0 or 5 if they exist
    if defm_stack.shape[0] > 1:
        plt.subplot(1, 3, 2)
        plt.imshow(tensor_to_image(defm_stack[0:1, ...]), cmap="gray")
        plt.title("Generated KL Image (index=0)")
        plt.axis("off")

        if defm_stack.shape[0] > 5:
            plt.subplot(1, 3, 3)
            plt.imshow(tensor_to_image(defm_stack[5:6, ...]), cmap="gray")
            plt.title("Generated KL Image (index=5)")
            plt.axis("off")

    plt.show()

if __name__ == "__main__":
    main()
