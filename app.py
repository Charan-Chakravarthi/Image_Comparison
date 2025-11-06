import streamlit as st
import cv2
import numpy as np
from PIL import Image
import imagehash
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.nn.functional import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

# Load pre-trained ResNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True).to(device)
model.eval()

# Define image transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_image_features(img):
    """Extract features using both traditional CV and deep learning approaches."""
    # Convert PIL image to OpenCV format for traditional features
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # 1. Deep Learning Features (ResNet)
    with torch.no_grad():
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        deep_features = model.forward(img_tensor)
        deep_features = deep_features.squeeze().cpu().numpy()
    
    # 2. Perceptual hash
    phash = imagehash.average_hash(img, hash_size=8)
    
    # 3. Color histogram
    hist = cv2.calcHist([img_cv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    # 4. SIFT features
    sift = cv2.SIFT_create()
    _, descs = sift.detectAndCompute(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), None)
    if descs is not None:
        descs = np.mean(descs, axis=0) if len(descs) > 0 else np.zeros(128)
    else:
        descs = np.zeros(128)
    
    return {
        'deep_features': deep_features,
        'phash': phash,
        'hist': hist,
        'sift': descs
    }

def calculate_similarity(features1, features2):
    """Calculate similarity using both deep learning and traditional features."""
    try:
        # 1. Deep Learning similarity
        deep_features1 = features1['deep_features'].reshape(1, -1)
        deep_features2 = features2['deep_features'].reshape(1, -1)
        deep_similarity = float(sklearn_cosine_similarity(deep_features1, deep_features2)[0][0])
        deep_similarity = (deep_similarity + 1) * 50
        
        # 2. Perceptual hash similarity
        hash_diff = float(features1['phash'] - features2['phash'])
        max_hash_diff = 64.0
        phash_similarity = 100.0 * (1.0 - hash_diff / max_hash_diff)
        
        # 3. Histogram similarity
        hist_similarity = float(cv2.compareHist(
            features1['hist'].reshape(-1, 1),
            features2['hist'].reshape(-1, 1),
            cv2.HISTCMP_CORREL
        ))
        hist_similarity = (hist_similarity + 1) * 50
        
        # 4. SIFT features similarity
        sift_diff = np.linalg.norm(features1['sift'] - features2['sift'])
        sift_similarity = max(0, 100 - (sift_diff * 100 / (np.sqrt(128) * 255)))
        
        # Ensure all similarities are in valid range
        deep_similarity = max(0, min(100, deep_similarity))
        phash_similarity = max(0, min(100, phash_similarity))
        hist_similarity = max(0, min(100, hist_similarity))
        sift_similarity = max(0, min(100, sift_similarity))
        
        # Weighted average
        weights = [0.4, 0.2, 0.2, 0.2]
        final_similarity = (
            weights[0] * deep_similarity +
            weights[1] * phash_similarity +
            weights[2] * hist_similarity +
            weights[3] * sift_similarity
        )
        
        return {
            'final': max(0, min(100, float(final_similarity))),
            'deep': float(deep_similarity),
            'phash': float(phash_similarity),
            'hist': float(hist_similarity),
            'sift': float(sift_similarity)
        }
        
    except Exception as e:
        st.error(f"Error in similarity calculation: {str(e)}")
        return {
            'final': 0.0,
            'deep': 0.0,
            'phash': 0.0,
            'hist': 0.0,
            'sift': 0.0
        }

def main():
    st.set_page_config(page_title="AI Image Comparison", layout="wide")
    
    st.title("AI Image Similarity Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("First Image")
        image1 = st.file_uploader("Upload first image", type=['jpg', 'jpeg', 'png'])
        if image1:
            st.image(image1, caption="Image 1", use_column_width=True)
    
    with col2:
        st.subheader("Second Image")
        image2 = st.file_uploader("Upload second image", type=['jpg', 'jpeg', 'png'])
        if image2:
            st.image(image2, caption="Image 2", use_column_width=True)
    
    if image1 and image2:
        if st.button("Compare Images"):
            with st.spinner("Comparing images..."):
                try:
                    # Process images
                    img1 = Image.open(image1).convert('RGB')
                    img2 = Image.open(image2).convert('RGB')
                    
                    # Get features
                    features1 = get_image_features(img1)
                    features2 = get_image_features(img2)
                    
                    # Calculate similarity
                    similarities = calculate_similarity(features1, features2)
                    
                    # Display results
                    st.subheader("Comparison Results")
                    
                    # Overall similarity with larger font
                    st.markdown(f"<h2 style='text-align: center; color: #4ecdc4;'>Overall Similarity: {similarities['final']:.2f}%</h2>", unsafe_allow_html=True)
                    
                    # Individual metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Deep Learning Similarity", f"{similarities['deep']:.2f}%")
                        st.metric("Perceptual Hash Similarity", f"{similarities['phash']:.2f}%")
                    
                    with col2:
                        st.metric("Color Histogram Similarity", f"{similarities['hist']:.2f}%")
                        st.metric("SIFT Features Similarity", f"{similarities['sift']:.2f}%")
                    
                except Exception as e:
                    st.error(f"Error processing images: {str(e)}")

if __name__ == '__main__':
    main()