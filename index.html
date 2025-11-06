import io
import os
from typing import Dict

import cv2
import imagehash
import numpy as np
import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

st.set_page_config(page_title="AI Image Similarity", layout="centered")


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load pretrained ResNet50
    model = models.resnet50(pretrained=True).to(device)
    model.eval()
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, preprocess, device


def get_image_features(img: Image.Image, model, preprocess, device) -> Dict:
    """Extract features: ResNet deep features, perceptual hash, color hist, SIFT descriptor summary."""
    # Convert PIL -> OpenCV BGR
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # 1) Deep features (ResNet)
    with torch.no_grad():
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        feats = model(img_tensor)
        deep_features = feats.squeeze().cpu().numpy()

    # 2) Perceptual hash (average hash, 8 -> 64 bits)
    phash = imagehash.average_hash(img, hash_size=8)

    # 3) Color histogram (8x8x8 bins) normalized
    hist = cv2.calcHist([img_cv], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # 4) SIFT (mean of descriptors or zeros)
    try:
        sift = cv2.SIFT_create()
        _, descs = sift.detectAndCompute(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), None)
        if descs is not None and len(descs) > 0:
            descs = np.mean(descs, axis=0)
        else:
            descs = np.zeros(128, dtype=np.float32)
    except Exception:
        # if SIFT not available for some reason, fallback to zeros
        descs = np.zeros(128, dtype=np.float32)

    return {
        "deep_features": deep_features,
        "phash": phash,
        "hist": hist,
        "sift": descs,
    }


def calculate_similarity(f1: Dict, f2: Dict) -> Dict:
    """Combine deep + phash + hist + sift into 0-100 metrics and weighted final score."""
    try:
        # deep: cosine similarity from -1..1 -> 0..100
        d1 = f1["deep_features"].reshape(1, -1)
        d2 = f2["deep_features"].reshape(1, -1)
        deep_sim = float(sklearn_cosine_similarity(d1, d2)[0][0])
        deep_sim = (deep_sim + 1.0) * 50.0

        # phash: hamming distance -> 0..100
        max_hash_diff = 64.0
        hash_diff = float(f1["phash"] - f2["phash"])
        phash_sim = 100.0 * (1.0 - (hash_diff / max_hash_diff))

        # hist: cv2.HISTCMP_CORREL returns -1..1 -> map to 0..100
        hist_sim = float(cv2.compareHist(f1["hist"].reshape(-1, 1),
                                         f2["hist"].reshape(-1, 1),
                                         cv2.HISTCMP_CORREL))
        hist_sim = (hist_sim + 1.0) * 50.0

        # sift: euclidean distance -> inverted score 0..100
        sift_diff = np.linalg.norm(f1["sift"] - f2["sift"])
        sift_sim = max(0.0, 100.0 - (sift_diff * 100.0 / (np.sqrt(128) * 255.0)))

        # clamp
        deep_sim = max(0.0, min(100.0, deep_sim))
        phash_sim = max(0.0, min(100.0, phash_sim))
        hist_sim = max(0.0, min(100.0, hist_sim))
        sift_sim = max(0.0, min(100.0, sift_sim))

        weights = [0.4, 0.2, 0.2, 0.2]
        final = (
            weights[0] * deep_sim +
            weights[1] * phash_sim +
            weights[2] * hist_sim +
            weights[3] * sift_sim
        )
        final = max(0.0, min(100.0, final))

        return {
            "final": final,
            "deep": deep_sim,
            "phash": phash_sim,
            "hist": hist_sim,
            "sift": sift_sim,
        }
    except Exception as e:
        st.error(f"Error computing similarity: {e}")
        return {"final": 0.0, "deep": 0.0, "phash": 0.0, "hist": 0.0, "sift": 0.0}


# ---------- Streamlit UI ----------
st.title("AI Image Similarity Comparison")

col1, col2 = st.columns(2)
uploaded1 = col1.file_uploader("Upload image 1", type=["png", "jpg", "jpeg"])
uploaded2 = col2.file_uploader("Upload image 2", type=["png", "jpg", "jpeg"])

st.markdown("---")
st.caption("Model: ResNet50 (pretrained). This runs on CPU if GPU not available.")

model, preprocess, device = load_model()

if uploaded1 is not None:
    try:
        img1 = Image.open(io.BytesIO(uploaded1.read())).convert("RGB")
        col1.image(img1, use_column_width=True, caption="Image 1 preview")
    except Exception as e:
        col1.error(f"Failed to read image 1: {e}")
        uploaded1 = None

if uploaded2 is not None:
    try:
        img2 = Image.open(io.BytesIO(uploaded2.read())).convert("RGB")
        col2.image(img2, use_column_width=True, caption="Image 2 preview")
    except Exception as e:
        col2.error(f"Failed to read image 2: {e}")
        uploaded2 = None

if st.button("Compare Images"):
    if uploaded1 is None or uploaded2 is None:
        st.warning("Please upload both images.")
    else:
        with st.spinner("Extracting features and computing similarity..."):
            # reload images from uploaded bytes (files were consumed above)
            img1 = Image.open(io.BytesIO(uploaded1.getvalue())).convert("RGB")
            img2 = Image.open(io.BytesIO(uploaded2.getvalue())).convert("RGB")

            f1 = get_image_features(img1, model, preprocess, device)
            f2 = get_image_features(img2, model, preprocess, device)

            sims = calculate_similarity(f1, f2)

        st.success(f"Overall Similarity: {sims['final']:.2f}%")
        st.metric(label="Deep (ML) similarity", value=f"{sims['deep']:.2f}%")
        st.metric(label="Perceptual hash similarity", value=f"{sims['phash']:.2f}%")
        st.metric(label="Color histogram similarity", value=f"{sims['hist']:.2f}%")
        st.metric(label="SIFT similarity", value=f"{sims['sift']:.2f}%")
        st.progress(int(round(sims["final"])))
        st.write("**Detailed breakdown**")
        st.write({
            "final": f"{sims['final']:.2f}%",
            "deep": f"{sims['deep']:.2f}%",
            "phash": f"{sims['phash']:.2f}%",
            "hist": f"{sims['hist']:.2f}%",
            "sift": f"{sims['sift']:.2f}%"
        })
