from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from PIL import Image
import imagehash
import io
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.nn.functional import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

app = Flask(__name__)

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
        print(f"Input tensor shape: {img_tensor.shape}")
        deep_features = model.forward(img_tensor)
        print(f"ResNet features shape: {deep_features.shape}")
        deep_features = deep_features.squeeze().cpu().numpy()
        print(f"Final features shape: {deep_features.shape}, Range: [{deep_features.min():.2f}, {deep_features.max():.2f}]")
    
    # 2. Perceptual hash (using a larger hash size for better accuracy)
    phash = imagehash.average_hash(img, hash_size=8)  # 8x8 = 64-bit hash
    print(f"Generated perceptual hash: {phash}")
    
    # 3. Color histogram
    hist = cv2.calcHist([img_cv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    # 4. SIFT features (simplified to top K keypoints)
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
        # 1. Deep Learning similarity (using cosine similarity)
        deep_features1 = features1['deep_features'].reshape(1, -1)
        deep_features2 = features2['deep_features'].reshape(1, -1)
        deep_similarity = float(sklearn_cosine_similarity(deep_features1, deep_features2)[0][0])
        # Convert from -1:1 range to 0:100 range
        deep_similarity = (deep_similarity + 1) * 50
        
        # 2. Perceptual hash similarity (0-100)
        try:
            # Calculate hamming distance between hashes and convert to percentage
            hash_diff = float(features1['phash'] - features2['phash'])
            max_hash_diff = 64.0  # ImageHash uses 64-bit hashes
            phash_similarity = 100.0 * (1.0 - hash_diff / max_hash_diff)
            print(f"Hash 1: {features1['phash']}")
            print(f"Hash 2: {features2['phash']}")
            print(f"Hash difference: {hash_diff}/{max_hash_diff} -> Similarity: {phash_similarity:.2f}%")
        except Exception as e:
            print(f"Error in phash calculation: {str(e)}")
            phash_similarity = 0.0
        
        # 3. Histogram similarity (0-100)
        hist_similarity = float(cv2.compareHist(
            features1['hist'].reshape(-1, 1),
            features2['hist'].reshape(-1, 1),
            cv2.HISTCMP_CORREL
        ))
        # Convert from -1:1 range to 0:100 range
        hist_similarity = (hist_similarity + 1) * 50
        
        # 4. SIFT features similarity (0-100)
        sift_diff = np.linalg.norm(features1['sift'] - features2['sift'])
        sift_similarity = max(0, 100 - (sift_diff * 100 / (np.sqrt(128) * 255)))  # 128 is SIFT descriptor size
        
        # Ensure all similarities are in valid range
        deep_similarity = max(0, min(100, deep_similarity))
        phash_similarity = max(0, min(100, phash_similarity))
        hist_similarity = max(0, min(100, hist_similarity))
        sift_similarity = max(0, min(100, sift_similarity))
        
        # Weighted average (adjustable weights)
        weights = [0.4, 0.2, 0.2, 0.2]  # deep learning, phash, histogram, SIFT
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
        print(f"Error in similarity calculation: {str(e)}")
        # Return safe default values in case of error
        return {
            'final': 0.0,
            'deep': 0.0,
            'phash': 0.0,
            'hist': 0.0,
            'sift': 0.0
        }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare_images():
    try:
        # Get images from request
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({
                'success': False,
                'message': "Both images are required"
            }), 400
            
        image1 = request.files['image1']
        image2 = request.files['image2']

        # Open and process images
        try:
            img1 = Image.open(io.BytesIO(image1.read())).convert('RGB')
            img2 = Image.open(io.BytesIO(image2.read())).convert('RGB')
        except Exception as e:
            return jsonify({
                'success': False,
                'message': "Invalid image format"
            }), 400

        # Get features
        try:
            features1 = get_image_features(img1)
            features2 = get_image_features(img2)
            print("Successfully extracted features")
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            raise

        # Calculate similarity
        try:
            # Use the combined ML and traditional similarity calculation
            similarities = calculate_similarity(features1, features2)
            print(f"Deep Learning similarity: {similarities['deep']:.2f}%")
            print(f"Individual similarities - Phash: {similarities['phash']:.2f}%, " +
                  f"Hist: {similarities['hist']:.2f}%, SIFT: {similarities['sift']:.2f}%")
            print(f"Final similarity score: {similarities['final']:.2f}%")
        except Exception as e:
            print(f"Error calculating similarity: {str(e)}")
            raise

        return jsonify({
            'success': True,
            'similarity': f"{similarities['final']:.2f}",
            'deep_similarity': f"{similarities['deep']:.2f}",
            'phash_similarity': f"{similarities['phash']:.2f}",
            'hist_similarity': f"{similarities['hist']:.2f}",
            'sift_similarity': f"{similarities['sift']:.2f}",
            'message': 'Comparison completed successfully'
        })
    except Exception as e:
        print(f"Error in comparison: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to compare images'
        }), 400

if __name__ == '__main__':
    app.run(debug=True)
