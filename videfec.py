import streamlit as st
import numpy as np
import io
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw

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
    shifted = np.stack([r, g, b], axis=-1)
    return Image.fromarray(shifted)

def apply_blur(image):
    image_rgb = image.convert('RGB')
    is_cartoon = detect_if_cartoon(image_rgb)
    if is_cartoon:
        return image_rgb.filter(ImageFilter.GaussianBlur(radius=3))
    else:
        return image_rgb.filter(ImageFilter.GaussianBlur(radius=7))

def apply_edge_detection(image):
    image_rgb = image.convert('RGB')
    edges = image_rgb.convert('L').filter(ImageFilter.FIND_EDGES)
    return edges.convert('RGB')

def apply_cartoon_effect(image):
    image = image.convert('RGB')
    gray = image.convert('L')
    blurred = gray.filter(ImageFilter.MedianFilter(size=5))
    edges = blurred.filter(ImageFilter.FIND_EDGES)
    color = image.filter(ImageFilter.SMOOTH)
    edge_image = edges.convert('RGB')
    cartoon = Image.blend(color, image, 0.3)
    return ImageOps.colorize(edges, black="white", white="black").convert('L')

def apply_enhanced_cartoon(image):
    """
    تأثير كرتوني محسّن يجمع بين تقليل الألوان وتعزيز الحواف
    """
    image = image.convert('RGB')
    img_array = np.array(image)
    img_array = (img_array // 32) * 32
    temp_img = Image.fromarray(img_array)
    enhancer = ImageEnhance.Color(temp_img)
    color_enhanced = enhancer.enhance(1.5)
    edges = image.filter(ImageFilter.CONTOUR)
    edges = edges.convert('L')
    threshold = 100
    edge_array = np.array(edges)
    edge_array = np.where(edge_array < threshold, 0, 255).astype(np.uint8)
    strong_edges = Image.fromarray(edge_array)
    color_array = np.array(color_enhanced)
    edge_mask = np.array(strong_edges.convert("L"))
    result_array = color_array.copy()
    edge_indices = edge_mask < 128
    for c in range(3):
        channel = result_array[:,:,c]
        channel[edge_indices] = 0
        result_array[:,:,c] = channel
    return Image.fromarray(result_array)

def apply_negative(image):
    image_rgb = image.convert('RGB')
    return ImageOps.invert(image_rgb)

def apply_sepia(image):
    image = np.array(image.convert('RGB'))
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    r_new = np.clip(r * 0.393 + g * 0.769 + b * 0.189, 0, 255).astype('uint8')
    g_new = np.clip(r * 0.349 + g * 0.686 + b * 0.168, 0, 255).astype('uint8')
    b_new = np.clip(r * 0.272 + g * 0.534 + b * 0.131, 0, 255).astype('uint8')
    sepia = np.stack([r_new, g_new, b_new], axis=-1)
    return Image.fromarray(sepia)

def apply_emboss(image):
    image_rgb = image.convert('RGB')
    is_cartoon = detect_if_cartoon(image_rgb)
    if is_cartoon:
        smoothed = image_rgb.filter(ImageFilter.SMOOTH)
        embossed = smoothed.filter(ImageFilter.EMBOSS)
        return Image.blend(embossed, image_rgb, 0.3)
    else:
        return image_rgb.filter(ImageFilter.EMBOSS)

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

def detect_if_cartoon(image):
    img_array = np.array(image.convert('RGB'))
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    unique_colors = len(np.unique(r)) + len(np.unique(g)) + len(np.unique(b))
    gray = image.convert('L')
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edges)
    edge_ratio = np.sum(edge_array > 128) / (edge_array.shape[0] * edge_array.shape[1])
    if unique_colors < 5000 and edge_ratio > 0.05:
        return True
    else:
        return False

def main():
    st.set_page_config(page_title="Image Effects Editor", layout="wide")
    st.title("🎭 Image Effects Editor")
    st.markdown("Enhance your images with stunning effects! Upload an image or GIF and choose an effect to apply.")
    
    uploaded_file = st.file_uploader("Upload an image or GIF", type=["png", "jpg", "jpeg", "gif"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
            effect = st.selectbox("Choose an effect:", [
                "Glitch Effect", "Noise", "Ghost Effect", "RGB Shift", "Blur",
                "Edge Detection", "Cartoon", "Enhanced Cartoon", "Negative", "Sepia", "Emboss"
            ])
            
            if effect in ["Blur", "Sepia", "Noise"]:
                intensity = st.slider("Effect Intensity", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
            else:
                intensity = 1.0
        
        with col2:
            try:
                if effect == "Glitch Effect":
                    result = apply_glitch_effect(image)
                elif effect == "Noise":
                    if intensity != 1.0:
                        image_array = np.array(image.convert('RGB'))
                        noise = np.random.randint(0, int(50 * intensity), image_array.shape, dtype='uint8')
                        noisy_image = np.clip(image_array + noise, 0, 255).astype('uint8')
                        result = Image.fromarray(noisy_image)
                    else:
                        result = apply_noise_effect(image)
                elif effect == "Ghost Effect":
                    result = apply_ghost_effect(image)
                elif effect == "RGB Shift":
                    result = apply_rgb_shift(image)
                elif effect == "Blur":
                    image_rgb = image.convert('RGB')
                    is_cartoon = detect_if_cartoon(image_rgb)
                    radius = 3 if is_cartoon else 7
                    radius = radius * intensity
                    result = image_rgb.filter(ImageFilter.GaussianBlur(radius=radius))
                elif effect == "Edge Detection":
                    result = apply_edge_detection(image)
                elif effect == "Cartoon":
                    result = apply_cartoon_effect(image)
                elif effect == "Enhanced Cartoon":
                    result = apply_enhanced_cartoon(image)
                elif effect == "Negative":
                    result = apply_negative(image)
                elif effect == "Sepia":
                    if intensity != 1.0:
                        image_array = np.array(image.convert('RGB'))
                        r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
                        intensity_factor = intensity
                        r_new = np.clip(r * 0.393 * intensity_factor + g * 0.769 * intensity_factor + b * 0.189 * intensity_factor, 0, 255).astype('uint8')
                        g_new = np.clip(r * 0.349 * intensity_factor + g * 0.686 * intensity_factor + b * 0.168 * intensity_factor, 0, 255).astype('uint8')
                        b_new = np.clip(r * 0.272 * intensity_factor + g * 0.534 * intensity_factor + b * 0.131 * intensity_factor, 0, 255).astype('uint8')
                        sepia = np.stack([r_new, g_new, b_new], axis=-1)
                        result = Image.fromarray(sepia)
                    else:
                        result = apply_sepia(image)
                elif effect == "Emboss":
                    result = apply_emboss(image)
                
                st.image(result, caption=f"Applied {effect}", use_container_width=True)
                
                img_byte_arr = io.BytesIO()
                result.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                st.download_button(
                    label="Download Edited Image",
                    data=img_byte_arr,
                    file_name=f"edited_{effect.lower().replace(' ', '_')}.png",
                    mime="image/png"
                )
                
                is_cartoon = detect_if_cartoon(image)
                st.info(f"Image detected as: {'Cartoon/Illustration' if is_cartoon else 'Natural Photo'}")
                
            except Exception as e:
                st.error(f"An error occurred while applying the effect: {e}")
                st.info("Please try a different effect or upload a different image.")

if __name__ == "__main__":
    main()