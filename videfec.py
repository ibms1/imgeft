import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

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
    noisy_image = np.clip(image + noise, 0, 255).astype('uint8')
    return Image.fromarray(noisy_image)

def apply_ghost_effect(image):
    image = np.array(image.convert('RGB'))
    alpha = 0.5
    ghost = (image * alpha + np.roll(image, 5, axis=1) * (1 - alpha)).astype('uint8')
    return Image.fromarray(ghost)

def apply_rgb_shift(image):
    image = np.array(image.convert('RGB'))
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    r = np.roll(r, 5, axis=1)
    b = np.roll(b, -5, axis=0)
    shifted = np.stack([b, g, r], axis=-1)
    return Image.fromarray(shifted)

def apply_blur(image):
    return image.filter(ImageFilter.GaussianBlur(radius=7))

def apply_edge_detection(image):
    return image.convert("L").filter(ImageFilter.FIND_EDGES)

def apply_cartoon_effect(image):
    image = image.convert('RGB')
    gray = image.convert('L')
    blurred = gray.filter(ImageFilter.MedianFilter(size=5))
    edges = blurred.filter(ImageFilter.FIND_EDGES)
    color = image.filter(ImageFilter.SMOOTH)
    cartoon = Image.composite(color, image, edges)
    return cartoon

def apply_negative(image):
    return Image.eval(image, lambda x: 255 - x)

def apply_sepia(image):
    image = np.array(image.convert('RGB'))
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    r_new = np.clip(r * 0.393 + g * 0.769 + b * 0.189, 0, 255).astype('uint8')
    g_new = np.clip(r * 0.349 + g * 0.686 + b * 0.168, 0, 255).astype('uint8')
    b_new = np.clip(r * 0.272 + g * 0.534 + b * 0.131, 0, 255).astype('uint8')
    sepia = np.stack([b_new, g_new, r_new], axis=-1)
    return Image.fromarray(sepia)

def apply_emboss(image):
    return image.filter(ImageFilter.EMBOSS)

def apply_tiktok_falling_effect(image):
    image = np.array(image.convert('RGB'))
    h, w = image.shape[:2]
    result = np.zeros_like(image)
    num_frames = 8
    max_shift = int(h * 0.15)
    for i in range(num_frames):
        shift = int((i / num_frames) * max_shift)
        y_src_start = 0
        y_src_end = h - shift
        y_dst_start = shift
        y_dst_end = h
        if y_src_end > y_src_start and y_dst_end > y_dst_start:
            src_region = image[y_src_start:y_src_end, :]
            alpha = 1.0 - (i / num_frames) * 0.8
            if i == 0:
                result[y_dst_start:y_dst_end, :] = src_region
            else:
                dst_region = result[y_dst_start:y_dst_end, :]
                result[y_dst_start:y_dst_end, :] = (dst_region * (1.0 - alpha) + src_region * alpha)
    return Image.fromarray(result.astype('uint8'))

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