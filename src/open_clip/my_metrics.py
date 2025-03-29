import torch
import torch.nn.functional as F
import os


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
#                               FROM CYCLIP
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


@torch.no_grad()
def itm_eval(text_embeddings, image_embeddings):

    # sim_matrix_i2t = image_embeddings @ text_embeddings.t()
    # sim_matrix_t2i = text_embeddings @ image_embeddings.t()

    ## Image -> Text
    # ranks = np.zeros(len(sim_matrix_i2t))
    ranks = np.zeros(len(image_embeddings))

    for index in range(0, len(image_embeddings), 5):
        scores = image_embeddings[index] @ text_embeddings.t()
        # scores = sim_matrix_i2t[index]
        li = np.argsort(scores.detach().cpu().numpy())[::-1]
        for i in range(len(li)):
            if index <= li[i] and li[i] <= index + 4:
                rank = i
                break
        ranks[index] = rank

        # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    ## Image -> Text
    ranks = np.zeros(len(text_embeddings))
    for index in range(len(text_embeddings)):
        scores = text_embeddings[index] @ image_embeddings.t()
        # for index, scores in tqdm(enumerate(sim_matrix_t2i)):
        scores = scores[::5]
        li = np.argsort(scores.detach().cpu().numpy())[::-1]
        for i in range(len(li)):
            if li[i] == index // 5:
                rank = i
                break
        ranks[index] = rank

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {
        "txt_r1": tr1,
        "txt_r5": tr5,
        "txt_r10": tr10,
        "txt_r_mean": tr_mean,
        "img_r1": ir1,
        "img_r5": ir5,
        "img_r10": ir10,
        "img_r_mean": ir_mean,
        "r_mean": r_mean,
    }

    return eval_result


def get_all_embeddings(
    model,
    all_texts,
    all_images,
    root,
    preprocess,
    tokenizer,
    batch_size=1024,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_embeddings = []
    image_embeddings = []

    with torch.no_grad():
        dataloader_texts = list(batch(all_texts, batch_size))
        dataloader_images = list(batch(all_images, batch_size))

        bar = zip(dataloader_texts, dataloader_images)
        bar = tqdm(bar, total=len(dataloader_texts), desc="Encoding batches")

        for texts, images in bar:
            # Tokenize text
            text_tokens = tokenizer(texts).to(device)

            # Preprocess and stack images
            image_tensors = torch.stack(
                [
                    preprocess(Image.open(os.path.join(root, img)).convert("RGB"))
                    for img in images
                ]
            ).to(device)

            # Encode
            image_embedding = model.encode_image(image_tensors)
            text_embedding = model.encode_text(text_tokens)

            # Normalize
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

            text_embeddings.append(text_embedding)
            image_embeddings.append(image_embedding)

        text_embeddings = torch.cat(text_embeddings)
        image_embeddings = torch.cat(image_embeddings)
        return text_embeddings, image_embeddings
