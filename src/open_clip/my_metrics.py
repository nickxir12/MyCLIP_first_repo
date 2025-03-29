import torch
import torch.nn.functional as F


def compute_consistency_score(model, dataloader, device):
    """
    Computes the Consistency Score for a given model and dataset.

    Args:
        model: The trained CyCLIP model.
        dataloader: DataLoader providing (image, text) pairs.
        device: Device to run the computations on ('cuda' or 'cpu').

    Returns:
        The average Consistency Score across the dataset.
    """
    model.eval()
    total_score = 0.0
    num_samples = 0

    with torch.no_grad():
        for images, texts in dataloader:
            images, texts = images.to(device), texts.to(device)

            # Encode images and texts
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

            # Normalize features
            image_features = F.normalize(image_features, p=2, dim=-1)
            text_features = F.normalize(text_features, p=2, dim=-1)

            # Compute cosine similarity
            similarities = torch.sum(image_features * text_features, dim=-1)

            # Accumulate scores
            total_score += similarities.sum().item()
            num_samples += images.size(0)

    # Average consistency score
    average_score = total_score / num_samples
    return average_score


#       USAGE EXAMPLE

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# consistency_score = compute_consistency_score(model, dataloader, device)
# print(f"Consistency Score: {consistency_score:.4f}")


import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import json
import pandas as pd
import matplotlib.pyplot as plt


def evaluate_model(
    model,
    preprocess,
    tokenizer,
    test_data,
    image_folder,
    device,
    top_k=5,
    captions_per_image=5,
):
    """
    Evaluates an image-text retrieval model using a preloaded test dataset.

    Args:
        model: The OpenCLIP model to use for encoding.
        preprocess: The preprocessing function for images.
        tokenizer: The tokenizer for text encoding.
        test_data (list): Preloaded test dataset (list of image-caption dictionaries).
        image_folder (str): Path to the folder containing Flickr30k images.
        device: PyTorch device (e.g., "cuda" or "cpu").
        top_k (int): Number of top retrieved captions/images to consider.
        captions_per_image (int): Number of captions associated with each image.

    Returns:
        DataFrame containing image paths, retrieved captions, and similarity scores.
    """

    # Store extracted features
    image_features = []
    text_features = []
    image_paths = []
    captions = []
    caption_to_image = {}

    # Process each test image
    for item in tqdm(test_data, desc="Extracting Features"):
        img_path = f"{image_folder}/{item['filename']}"
        image_paths.append(img_path)

        # Encode image
        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            img_feat = model.encode_image(image)
        image_features.append(img_feat.cpu().numpy())

        # Encode captions
        for sentence in item["sentences"]:
            text = sentence["raw"]
            text_tokenized = tokenizer([text]).to(device)

            with torch.no_grad():
                text_feat = model.encode_text(text_tokenized)

            text_features.append(text_feat.cpu().numpy())
            captions.append(text)
            caption_to_image[text] = (
                img_path  # Store which image the caption belongs to
            )

    # Convert to numpy arrays
    image_features = np.vstack(image_features)
    text_features = np.vstack(text_features)

    print(
        f"✅ Extracted {image_features.shape[0]} image features and {text_features.shape[0]} text features."
    )

    # Normalize features
    image_features = image_features / np.linalg.norm(
        image_features, axis=1, keepdims=True
    )
    text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)

    # Compute similarity matrix (dot product)
    similarity_matrix = np.dot(text_features, image_features.T)

    # Compute retrieval accuracy
    def evaluate_retrieval(similarity_matrix, top_k=1, captions_per_image=5):
        N_captions, N_images = similarity_matrix.shape
        assert (
            N_captions == N_images * captions_per_image
        ), f"Number of captions ({N_captions}) must be {captions_per_image} times the number of images ({N_images})."

        top_k_indices = np.argsort(similarity_matrix, axis=1)[:, -top_k:]
        correct = 0
        for caption_idx in range(N_captions):
            correct_image_idx = caption_idx // captions_per_image
            if correct_image_idx in top_k_indices[caption_idx]:
                correct += 1
        return correct / N_captions

    # Compute accuracy for top-1, top-5, top-10
    for k in [1, 5, 10]:
        acc = evaluate_retrieval(
            similarity_matrix, top_k=k, captions_per_image=captions_per_image
        )
        print(f"Top-{k} Accuracy: {acc * 100:.2f}%")

    # Retrieve top-k captions per image
    top_k_indices = np.argsort(similarity_matrix, axis=0)[
        -top_k:
    ].T  # Sorting and taking top-k

    # Store results
    results = []
    for img_idx, indices in enumerate(top_k_indices):
        img_path = image_paths[img_idx]
        retrieved_captions = [captions[idx] for idx in indices[::-1]]
        retrieved_probs = [similarity_matrix[idx, img_idx] for idx in indices[::-1]]

        results.append(
            {
                "Image": img_path,
                "Top-5 Matches": list(zip(retrieved_captions, retrieved_probs)),
            }
        )

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    return df_results
