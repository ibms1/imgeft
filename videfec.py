import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance

def apply_glitch_effect(image):
    image = np.array(image)
    h, w = image.shape[:2]
    glitch = image.copy()
    
    # Check if image is RGB
    if len(image.shape) == 3:
        glitch[:, :w//3, 0] = np.roll(glitch[:, :w//3, 0], 10, axis=0)
        glitch[:, w//3:2*w//3, 1] = np.roll(glitch[:, w//3:2*w//3, 1], -10, axis=0)
        glitch[:, 2*w//3:, 2] = np.roll(glitch[:, 2*w//3:, 2], 15, axis=0)
    
    return Image.fromarray(glitch)

def apply_noise_effect(image):
    image = np.array(image)
    noise = np.random.randint(0, 50, image.shape, dtype='uint8')
    noisy_image = cv2.add(image, noise)
    return Image.fromarray(noisy_image)

def apply_ghost_effect(image):
    image = np.array(image)
    alpha = 0.5
    ghost = cv2.addWeighted(image, alpha, np.roll(image, 5, axis=1), 1 - alpha, 0)
    return Image.fromarray(ghost)

def apply_rgb_shift(image):
    # Convert to RGB if not already
    pil_image = image.convert('RGB')
    image = np.array(pil_image)
    
    # Split channels and apply shift
    b, g, r = cv2.split(image)  # OpenCV uses BGR order
    r = np.roll(r, 5, axis=1)
    b = np.roll(b, -5, axis=0)
    
    # Merge channels back
    shifted = cv2.merge((b, g, r))
    return Image.fromarray(shifted)

def apply_blur(image):
    image = np.array(image)
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    return Image.fromarray(blurred)

def apply_edge_detection(image):
    image = np.array(image.convert("L"))
    edges = cv2.Canny(image, 100, 200)
    return Image.fromarray(edges)

def apply_cartoon_effect(image):
    # Convert to RGB if not already
    pil_image = image.convert('RGB')
    image = np.array(pil_image)
    
    # Convert BGR to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(image, 9, 250, 250)
    
    # Convert grayscale edges to 3 channel for bitwise_and
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    cartoon = cv2.bitwise_and(color, edges_rgb)
    
    return Image.fromarray(cartoon)

def apply_negative(image):
    image = np.array(image)
    negative = cv2.bitwise_not(image)
    return Image.fromarray(negative)

def apply_sepia(image):
    # Convert to RGB if not already
    pil_image = image.convert('RGB')
    image = np.array(pil_image)
    
    # OpenCV sepia filter
    sepia_kernel = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ])
    
    # Manual transformation
    b, g, r = cv2.split(image)
    b_new = np.clip(r * 0.272 + g * 0.534 + b * 0.131, 0, 255).astype(np.uint8)
    g_new = np.clip(r * 0.349 + g * 0.686 + b * 0.168, 0, 255).astype(np.uint8)
    r_new = np.clip(r * 0.393 + g * 0.769 + b * 0.189, 0, 255).astype(np.uint8)
    
    sepia = cv2.merge([b_new, g_new, r_new])
    return Image.fromarray(sepia)

def apply_emboss(image):
    image = np.array(image)
    kernel = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
    embossed = cv2.filter2D(image, -1, kernel)
    return Image.fromarray(embossed)

def apply_tiktok_face_duplication(image):
    # Convert to RGB
    pil_image = image.convert('RGB')
    img = np.array(pil_image)
    
    # Create a copy of the image to work with
    result = img.copy()
    
    # Get dimensions
    h, w = img.shape[:2]
    
    # Create a duplicate face effect - top half normal, bottom half duplicated and offset
    # Dividing point (roughly where neck would be in a portrait)
    split_point = int(h * 0.6)
    
    # Take the top portion of the face from the original image
    top_half = img[:split_point, :].copy()
    
    # Take a portion of the face to duplicate (typically eyes/nose region)
    face_portion = img[int(h*0.2):int(h*0.5), :].copy()
    
    # Place the duplicated face portion lower in the image with some transparency
    alpha = 0.7  # Transparency factor
    
    # Calculate destination region
    dest_y1 = split_point
    dest_y2 = min(dest_y1 + face_portion.shape[0], h)
    
    # If the region fits
    if dest_y2 > dest_y1:
        # Get the height of the region that will fit
        region_h = dest_y2 - dest_y1
        
        # Create a region for blending
        result[dest_y1:dest_y2, :] = cv2.addWeighted(
            result[dest_y1:dest_y2, :], 1-alpha,
            face_portion[:region_h, :], alpha, 0
        )
    
    return Image.fromarray(result)

def main():
    st.set_page_config(page_title="Image Effects Editor", layout="wide")
    st.title("🎭 Image Effects Editor")
    st.markdown("Enhance your images with stunning effects! Upload an image or GIF and select an effect to apply.")
    
    uploaded_file = st.file_uploader("Upload an Image or GIF", type=["png", "jpg", "jpeg", "gif"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        effect = st.selectbox("Choose an Effect:", [
            "Glitch Effect", "Noise", "Ghost Effect", "RGB Shift", "Blur",
            "Edge Detection", "Cartoon", "Negative", "Sepia", "Emboss", "TikTok Face Duplication"
        ])
        
        if effect == "Glitch Effect":
            result = apply_glitch_effect(image)
        elif effect == "Noise":
            result = apply_noise_effect(image)
        elif effect == "Ghost Effect":
            result = apply_ghost_effect(image)
        elif effect == "RGB Shift":
            result = apply_rgb_shift(image)
        elif effect == "Blur":
            result = apply_blur(image)
        elif effect == "Edge Detection":
            result = apply_edge_detection(image)
        elif effect == "Cartoon":
            result = apply_cartoon_effect(image)
        elif effect == "Negative":
            result = apply_negative(image)
        elif effect == "Sepia":
            result = apply_sepia(image)
        elif effect == "Emboss":
            result = apply_emboss(image)
        elif effect == "TikTok Face Duplication":
            result = apply_tiktok_face_duplication(image)
        
        # Fixed deprecated parameter
        st.image(result, caption=f"{effect} Applied", use_container_width=True)

if __name__ == "__main__":
    main()