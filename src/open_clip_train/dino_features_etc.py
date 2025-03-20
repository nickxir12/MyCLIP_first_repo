from transformers import AutoImageProcessor, AutoModel
import torch.nn.functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_dino_model(model_name="facebook/dinov2-small", device=device):
    """Load the pre-trained DinoV2 model."""
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    return processor, model


def extract_dino_features(images, dino_model, dino_processor, device=device):
    # Normalize images to [0, 1] range
    images = (images - images.min()) / (images.max() - images.min())

    inputs = dino_processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        features = dino_model(**inputs).last_hidden_state[:, 0, :]  # Use CLS token
    return features


def compute_pairwise_similarities(features):
    # Normalize features for cosine similarity
    normalized_features = F.normalize(features, dim=1)
    similarities = torch.mm(
        normalized_features, normalized_features.T
    )  # Pairwise cosine similarity
    return similarities


def create_soft_labels(similarities, temperature=0.02):
    """Convert similarities into probabilities using softmax."""
    soft_labels = F.softmax(similarities / temperature, dim=1)
    return soft_labels


def compute_soft_label_loss(predicted_similarities, soft_labels):
    """Compute KL divergence between predicted similarities and soft labels."""
    loss = F.kl_div(
        F.log_softmax(predicted_similarities, dim=1), soft_labels, reduction="batchmean"
    )
    return loss
