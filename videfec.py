import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance

def apply_glitch_effect(image):
    image = np.array(image.convert('RGB'))
    h, w = image.shape[:2]
    glitch = image.copy()
    glitch[:, :w//3, 0] = np.roll(glitch[:, :w//3, 0], 10, axis=0)
    glitch[:, w//3:2*w//3, 1] = np.roll(glitch[:, w//3:2*w//3, 1], -10, axis=0)
    glitch[:, 2*w//3:, 2] = np.roll(glitch[:, 2*w//3:, 2], 15, axis=0)
    return Image.fromarray(glitch)

def apply_noise_effect(image):
    image = np.array(image.convert('RGB'))
    noise = np.random.randint(0, 50, image.shape, dtype='uint8')
    noisy_image = cv2.add(image, noise)
    return Image.fromarray(noisy_image)

def apply_ghost_effect(image):
    image = np.array(image.convert('RGB'))
    alpha = 0.5
    ghost = cv2.addWeighted(image, alpha, np.roll(image, 5, axis=1), 1 - alpha, 0)
    return Image.fromarray(ghost)

def apply_rgb_shift(image):
    pil_image = image.convert('RGB')
    image = np.array(pil_image)
    b, g, r = cv2.split(image)
    r = np.roll(r, 5, axis=1)
    b = np.roll(b, -5, axis=0)
    shifted = cv2.merge((b, g, r))
    return Image.fromarray(shifted)

def apply_blur(image):
    image = np.array(image.convert('RGB'))
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    return Image.fromarray(blurred)

def apply_edge_detection(image):
    image = np.array(image.convert("L"))
    edges = cv2.Canny(image, 100, 200)
    return Image.fromarray(edges)

def apply_cartoon_effect(image):
    pil_image = image.convert('RGB')
    image = np.array(pil_image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(image, 9, 250, 250)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    cartoon = cv2.bitwise_and(color, edges_rgb)
    return Image.fromarray(cartoon)

def apply_negative(image):
    image = np.array(image.convert('RGB'))
    negative = cv2.bitwise_not(image)
    return Image.fromarray(negative)

def apply_sepia(image):
    pil_image = image.convert('RGB')
    image = np.array(pil_image)
    b, g, r = cv2.split(image)
    b_new = np.clip(r * 0.272 + g * 0.534 + b * 0.131, 0, 255).astype(np.uint8)
    g_new = np.clip(r * 0.349 + g * 0.686 + b * 0.168, 0, 255).astype(np.uint8)
    r_new = np.clip(r * 0.393 + g * 0.769 + b * 0.189, 0, 255).astype(np.uint8)
    sepia = cv2.merge([b_new, g_new, r_new])
    return Image.fromarray(sepia)

def apply_emboss(image):
    image = np.array(image.convert('RGB'))
    kernel = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
    embossed = cv2.filter2D(image, -1, kernel)
    return Image.fromarray(embossed)

def apply_tiktok_falling_effect(image):
    pil_image = image.convert('RGB')
    img = np.array(pil_image)
    h, w = img.shape[:2]
    result = np.zeros_like(img)
    num_frames = 8
    max_shift = int(h * 0.15)
    for i in range(num_frames):
        shift = int((i / num_frames) * max_shift)
        y_src_start = 0
        y_src_end = h - shift
        y_dst_start = shift
        y_dst_end = h
        if y_src_end > y_src_start and y_dst_end > y_dst_start:
            src_region = img[y_src_start:y_src_end, :]
            alpha = 1.0 - (i / num_frames) * 0.8
            if i == 0:
                result[y_dst_start:y_dst_end, :] = src_region
            else:
                dst_region = result[y_dst_start:y_dst_end, :]
                result[y_dst_start:y_dst_end, :] = cv2.addWeighted(dst_region, 1.0, src_region, alpha, 0)
    kernel_motion_blur = np.zeros((15, 15))
    kernel_motion_blur[7, :] = np.ones(15)
    kernel_motion_blur = kernel_motion_blur / 15
    result = cv2.filter2D(result, -1, kernel_motion_blur)
    return Image.fromarray(result)

def main():
    st.set_page_config(page_title="Image Effects Editor", layout="wide")
    st.title("🎭 Image Effects Editor")
    st.markdown("Enhance your images with stunning effects! Upload an image or GIF and choose an effect to apply.")
    
    uploaded_file = st.file_uploader("Upload an image or GIF", type=["png", "jpg", "jpeg", "gif"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        # تقسيم الواجهة إلى عمودين
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
            effect = st.selectbox("Choose an effect:", [
                "Glitch Effect", "Noise", "Ghost Effect", "RGB Shift", "Blur",
                "Edge Detection", "Cartoon", "Negative", "Sepia", "Emboss", "Falling Effect"
            ])
        
        with col2:
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
            elif effect == "Falling Effect":
                result = apply_tiktok_falling_effect(image)
            
            st.image(result, caption=f"Applied {effect}", use_container_width=True)
            
            # زر تحميل الصورة المعدلة
            result_bytes = result.tobytes()
            st.download_button(
                label="Download Edited Image",
                data=result_bytes,
                file_name=f"edited_{effect.lower().replace(' ', '_')}.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()