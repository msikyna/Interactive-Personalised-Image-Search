"""
CLIP service for text-to-vector and image-to-vector conversion.
Uses ViT-L/14 model for 768-dimensional embeddings.
Integrates CLIP model for similarity search with text queries.
"""

import sys
import os
import urllib.error
from PIL import Image
import numpy as np
from . import config

clip_surgery_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'XAI',
    'XAI models',
    'CLIP Surgery'
)
sys.path.insert(0, clip_surgery_path)

torch = None
clip = None


def _load_clip_runtime():
    global torch, clip

    if torch is not None and clip is not None:
        return torch, clip

    config.probe_native_import('torch', 'PyTorch')

    import torch as torch_module
    import clip as clip_module

    torch = torch_module
    clip = clip_module
    return torch, clip


class CLIPService:
    """Handle CLIP model operations for text and image encoding using ViT-L/14."""

    def __init__(self):
        """Initialize CLIP service with ViT-L/14 model."""
        self.device = os.environ.get('SIMSEARCH_CLIP_DEVICE', '').strip() or None
        default_clip_dir = '/home/xsikyna/clip'
        default_model_path = os.path.join(default_clip_dir, 'ViT-L-14.pt')

        self.model_name = os.environ.get('SIMSEARCH_CLIP_MODEL_NAME', 'ViT-L/14')
        self.download_root = os.environ.get('SIMSEARCH_CLIP_CACHE_DIR', default_clip_dir)
        # Default to fixed server path; can still be overridden by env var.
        self.model_path = os.environ.get('SIMSEARCH_CLIP_MODEL_PATH', default_model_path)

        self.model = None
        self.preprocess = None
        self.model_loaded = False

    def load_model(self):
        """Load the ViT-L/14 CLIP model and preprocessing pipeline."""
        if not self.model_loaded:
            try:
                torch_module, clip_module = _load_clip_runtime()
                if self.device is None:
                    self.device = "cuda" if torch_module.cuda.is_available() else "cpu"

                if self.model_path:
                    if not os.path.isfile(self.model_path):
                        raise RuntimeError(
                            f"CLIP model file not found at {self.model_path}. "
                            f"Place ViT-L-14.pt there or set SIMSEARCH_CLIP_MODEL_PATH."
                        )
                    print(f"Loading CLIP model from local file: {self.model_path} on {self.device}...")
                    self.model, self.preprocess = clip_module.load(self.model_path, device=self.device)
                else:
                    print(f"Loading CLIP model: {self.model_name} on {self.device}...")
                    self.model, self.preprocess = clip_module.load(
                        self.model_name,
                        device=self.device,
                        download_root=self.download_root
                    )

                self.model.eval()  # Set to evaluation mode
                self.model_loaded = True
                print(f"CLIP model loaded successfully on {self.device}")
                print(f"Native CLIP dimension: 768")
            except urllib.error.URLError as e:
                cache_dir = self.download_root or os.path.expanduser("~/.cache/clip")
                expected_file = "ViT-L-14.pt" if self.model_name == "ViT-L/14" else "model file"
                raise RuntimeError(
                    f"Failed to download CLIP weights ({e}). "
                    f"This server appears to be offline. "
                    f"Place {expected_file} into {cache_dir} or set SIMSEARCH_CLIP_MODEL_PATH "
                    f"to a local .pt file."
                ) from e

        return self.model_loaded


    def text_to_vector(self, text, normalize=True):
        """
        Convert text to feature vector using CLIP.

        Args:
            text: String or list of strings to encode
            normalize: Whether to L2-normalize embeddings

        Returns:
            numpy array: 768-dimensional feature vector(s)
        """
        if not self.model_loaded:
            self.load_model()

        # Handle single string or list of strings
        if isinstance(text, str):
            text = [text]

        torch_module, clip_module = _load_clip_runtime()

        # Tokenize text
        text_tokens = clip_module.tokenize(text).to(self.device)

        # Extract features
        with torch_module.no_grad():
            text_features = self.model.encode_text(text_tokens)

        if normalize:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        # Convert to numpy
        result = text_features.cpu().numpy()


        # Return single vector if single input, otherwise return all
        return result[0] if len(text) == 1 else result

    def image_to_vector(self, image_path):
        """
        Convert image to feature vector using CLIP.

        Args:
            image_path: Path to image file or PIL Image object

        Returns:
            numpy array: Normalized 768-dimensional feature vector
        """
        if not self.model_loaded:
            self.load_model()

        # Load image if path is provided
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path

        # Preprocess image
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        # Extract features
        torch_module, _ = _load_clip_runtime()
        with torch_module.no_grad():
            image_features = self.model.encode_image(image_tensor)

        # CLIP Surgery returns all tokens [batch, num_tokens, dim]
        # Extract only the class token (first token) which is the global image embedding
        if len(image_features.shape) == 3:
            image_features = image_features[:, 0, :]  # Take only class token

        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Convert to numpy
        return image_features.cpu().numpy()[0]

    def batch_text_to_vectors(self, texts, batch_size=32):
        """
        Convert multiple texts to vectors in batches.

        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once

        Returns:
            numpy array: Array of normalized 768-dimensional feature vectors
        """
        if not self.model_loaded:
            self.load_model()

        all_features = []
        torch_module, clip_module = _load_clip_runtime()

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            text_tokens = clip_module.tokenize(batch).to(self.device)

            with torch_module.no_grad():
                batch_features = self.model.encode_text(text_tokens)

            batch_features /= batch_features.norm(dim=-1, keepdim=True)
            all_features.append(batch_features.cpu().numpy())

        return np.vstack(all_features)

    def batch_images_to_vectors(self, image_paths, batch_size=32):
        """
        Convert multiple images to vectors in batches.

        Args:
            image_paths: List of image file paths or PIL Images
            batch_size: Number of images to process at once

        Returns:
            numpy array: Array of normalized 768-dimensional feature vectors
        """
        if not self.model_loaded:
            self.load_model()

        all_features = []
        torch_module, _ = _load_clip_runtime()

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []

            for img_path in batch_paths:
                if isinstance(img_path, str):
                    image = Image.open(img_path).convert('RGB')
                else:
                    image = img_path
                batch_tensors.append(self.preprocess(image))

            batch_tensor = torch_module.stack(batch_tensors).to(self.device)

            with torch_module.no_grad():
                batch_features = self.model.encode_image(batch_tensor)

            # CLIP Surgery returns all tokens [batch, num_tokens, dim]
            # Extract only the class token (first token) which is the global image embedding
            if len(batch_features.shape) == 3:
                batch_features = batch_features[:, 0, :]  # Take only class token

            batch_features /= batch_features.norm(dim=-1, keepdim=True)
            all_features.append(batch_features.cpu().numpy())

        return np.vstack(all_features)

    def compute_similarity(self, vector1, vector2):
        """
        Compute cosine similarity between two vectors.

        Args:
            vector1: First feature vector
            vector2: Second feature vector

        Returns:
            float: Cosine similarity score (0 to 1, higher is more similar)
        """
        # Ensure vectors are numpy arrays
        v1 = np.array(vector1)
        v2 = np.array(vector2)

        # Compute dot product (vectors are already normalized)
        similarity = np.dot(v1, v2)

        return float(similarity)


# Global instance - uses ViT-L/14 model (768-dimensional)
clip_service = CLIPService()
