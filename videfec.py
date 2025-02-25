import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

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
    shifted = np.stack([r, g, b], axis=-1)  # تصحيح ترتيب القنوات RGB
    return Image.fromarray(shifted)

def apply_blur(image):
    # تحسين وظيفة البلور لتعمل مع جميع أنواع الصور
    image_rgb = image.convert('RGB')
    
    # تطبيق تأثير البلور مع إعدادات مخصصة للصور الكرتونية
    # استخدام قيمة radius أقل للصور الكرتونية
    is_cartoon = detect_if_cartoon(image_rgb)
    
    if is_cartoon:
        # استخدام بلور أخف للصور الكرتونية
        return image_rgb.filter(ImageFilter.GaussianBlur(radius=3))
    else:
        # استخدام بلور أقوى للصور العادية
        return image_rgb.filter(ImageFilter.GaussianBlur(radius=7))

def apply_edge_detection(image):
    # تحويل الصورة إلى RGB قبل المعالجة
    image_rgb = image.convert('RGB')
    # تطبيق تأثير كشف الحواف
    edges = image_rgb.convert('L').filter(ImageFilter.FIND_EDGES)
    # تحويل النتيجة إلى RGB لضمان التوافق
    return edges.convert('RGB')

def apply_cartoon_effect(image):
    image = image.convert('RGB')
    gray = image.convert('L')
    blurred = gray.filter(ImageFilter.MedianFilter(size=5))
    edges = blurred.filter(ImageFilter.FIND_EDGES)
    color = image.filter(ImageFilter.SMOOTH)
    # تحسين تأثير الكرتون باستخدام تقنية أفضل للدمج
    edge_image = edges.convert('RGB')
    cartoon = Image.blend(color, image, 0.3)
    # استخدام الحواف كقناع
    return ImageOps.colorize(edges, black="white", white="black").convert('L')

def apply_negative(image):
    # تحسين وظيفة النيجاتيف لتعمل مع جميع أنواع الصور
    # استخدام ImageOps.invert بدلاً من Image.eval
    image_rgb = image.convert('RGB')
    return ImageOps.invert(image_rgb)

def apply_sepia(image):
    image = np.array(image.convert('RGB'))
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    r_new = np.clip(r * 0.393 + g * 0.769 + b * 0.189, 0, 255).astype('uint8')
    g_new = np.clip(r * 0.349 + g * 0.686 + b * 0.168, 0, 255).astype('uint8')
    b_new = np.clip(r * 0.272 + g * 0.534 + b * 0.131, 0, 255).astype('uint8')
    sepia = np.stack([r_new, g_new, b_new], axis=-1)  # تصحيح ترتيب القنوات RGB
    return Image.fromarray(sepia)

def apply_emboss(image):
    # تحسين وظيفة Emboss لتعمل مع جميع أنواع الصور
    image_rgb = image.convert('RGB')
    
    # التعرف على ما إذا كانت الصورة كرتونية
    is_cartoon = detect_if_cartoon(image_rgb)
    
    if is_cartoon:
        # تطبيق emboss مع معالجة خاصة للصور الكرتونية
        # أولاً تطبيق تأثير تنعيم بسيط
        smoothed = image_rgb.filter(ImageFilter.SMOOTH)
        # ثم تطبيق تأثير emboss بشكل معتدل
        embossed = smoothed.filter(ImageFilter.EMBOSS)
        # مزج النتيجة مع الصورة الأصلية للحفاظ على بعض التفاصيل
        return Image.blend(embossed, image_rgb, 0.3)
    else:
        # تطبيق تأثير emboss العادي للصور الطبيعية
        return image_rgb.filter(ImageFilter.EMBOSS)

def detect_if_cartoon(image):
    """
    وظيفة للكشف إذا كانت الصورة كرتونية أم لا
    الاستراتيجية: الصور الكرتونية عادة ما يكون لها عدد ألوان أقل وحواف أكثر وضوحاً
    """
    # تحويل الصورة إلى مصفوفة numpy
    img_array = np.array(image.convert('RGB'))
    
    # 1. فحص تنوع الألوان (الصور الكرتونية عادة ما يكون لها عدد ألوان أقل)
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    unique_colors = len(np.unique(r)) + len(np.unique(g)) + len(np.unique(b))
    
    # 2. فحص حدة الحواف (الصور الكرتونية عادة ما يكون لها حواف أكثر وضوحاً)
    gray = image.convert('L')
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edges)
    edge_ratio = np.sum(edge_array > 128) / (edge_array.shape[0] * edge_array.shape[1])
    
    # 3. اتخاذ القرار بناءً على معايير متعددة
    if unique_colors < 5000 and edge_ratio > 0.05:
        return True
    else:
        return False

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
                "Edge Detection", "Cartoon", "Negative", "Sepia", "Emboss"
            ])
        
        with col2:
            try:
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
                
                st.image(result, caption=f"Applied {effect}", use_container_width=True)
                
                # تحويل الصورة إلى بايتات للتنزيل
                img_byte_arr = io.BytesIO()
                result.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # زر تحميل الصورة المعدلة
                st.download_button(
                    label="Download Edited Image",
                    data=img_byte_arr,
                    file_name=f"edited_{effect.lower().replace(' ', '_')}.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"An error occurred while applying the effect: {e}")
                st.info("Please try a different effect or upload a different image.")

if __name__ == "__main__":
    import io
    main()