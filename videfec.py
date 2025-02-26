import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance

def apply_glitch_effect(image):
    # تحويل الصورة إلى مصفوفة numpy
    image = np.array(image.convert('RGB'))
    h, w = image.shape[:2]
    glitch = image.copy()
    
    # تطبيق تأثير الخلل
    glitch[:, :w//3, 0] = np.roll(glitch[:, :w//3, 0], 10, axis=0)
    glitch[:, w//3:2*w//3, 1] = np.roll(glitch[:, w//3:2*w//3, 1], -10, axis=0)
    glitch[:, 2*w//3:, 2] = np.roll(glitch[:, 2*w//3:, 2], 15, axis=0)
    
    return Image.fromarray(glitch)

def apply_noise_effect(image):
    # تحويل الصورة إلى مصفوفة numpy
    image = np.array(image.convert('RGB'))
    noise = np.random.randint(0, 50, image.shape, dtype='uint8')
    noisy_image = cv2.add(image, noise)
    return Image.fromarray(noisy_image)

def apply_ghost_effect(image):
    # تحويل الصورة إلى مصفوفة numpy
    image = np.array(image.convert('RGB'))
    alpha = 0.5
    ghost = cv2.addWeighted(image, alpha, np.roll(image, 5, axis=1), 1 - alpha, 0)
    return Image.fromarray(ghost)

def apply_rgb_shift(image):
    # تحويل الصورة إلى RGB
    pil_image = image.convert('RGB')
    image = np.array(pil_image)
    
    # فصل القنوات وتطبيق الإزاحة
    b, g, r = cv2.split(image)  # OpenCV يستخدم ترتيب BGR
    r = np.roll(r, 5, axis=1)
    b = np.roll(b, -5, axis=0)
    
    # دمج القنوات مرة أخرى
    shifted = cv2.merge((b, g, r))
    return Image.fromarray(shifted)

def apply_blur(image):
    # تحويل الصورة إلى مصفوفة numpy
    image = np.array(image.convert('RGB'))
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    return Image.fromarray(blurred)

def apply_edge_detection(image):
    # تحويل الصورة إلى مصفوفة numpy
    image = np.array(image.convert("L"))
    edges = cv2.Canny(image, 100, 200)
    return Image.fromarray(edges)

def apply_cartoon_effect(image):
    # تحويل الصورة إلى RGB
    pil_image = image.convert('RGB')
    image = np.array(pil_image)
    
    # تحويل BGR إلى رمادي
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(image, 9, 250, 250)
    
    # تحويل حواف الصورة الرمادية إلى 3 قنوات للعملية الثنائية
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    cartoon = cv2.bitwise_and(color, edges_rgb)
    
    return Image.fromarray(cartoon)

def apply_negative(image):
    # تحويل الصورة إلى RGB ثم إلى مصفوفة numpy
    image = np.array(image.convert('RGB'))
    negative = cv2.bitwise_not(image)
    return Image.fromarray(negative)

def apply_sepia(image):
    # تحويل الصورة إلى RGB
    pil_image = image.convert('RGB')
    image = np.array(pil_image)
    
    # تطبيق مرشح السيبيا
    b, g, r = cv2.split(image)
    b_new = np.clip(r * 0.272 + g * 0.534 + b * 0.131, 0, 255).astype(np.uint8)
    g_new = np.clip(r * 0.349 + g * 0.686 + b * 0.168, 0, 255).astype(np.uint8)
    r_new = np.clip(r * 0.393 + g * 0.769 + b * 0.189, 0, 255).astype(np.uint8)
    
    sepia = cv2.merge([b_new, g_new, r_new])
    return Image.fromarray(sepia)

def apply_emboss(image):
    # تحويل الصورة إلى RGB ثم إلى مصفوفة numpy
    image = np.array(image.convert('RGB'))
    kernel = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
    embossed = cv2.filter2D(image, -1, kernel)
    return Image.fromarray(embossed)

def apply_tiktok_falling_effect(image):
    # تحويل الصورة إلى RGB
    pil_image = image.convert('RGB')
    img = np.array(pil_image)
    
    # الحصول على أبعاد الصورة
    h, w = img.shape[:2]
    
    # إنشاء صورة النتيجة
    result = np.zeros_like(img)
    
    # تحديد عدد الإطارات وعامل الحركة
    num_frames = 8
    max_shift = int(h * 0.15)  # مقدار السقوط كنسبة من ارتفاع الصورة
    
    # إنشاء تأثير السقوط للأمام بعدة طبقات من الصورة
    for i in range(num_frames):
        # حساب مقدار الإزاحة للطبقة الحالية
        shift = int((i / num_frames) * max_shift)
        
        # تحديد مناطق الصورة الأصلية والهدف
        y_src_start = 0
        y_src_end = h - shift
        y_dst_start = shift
        y_dst_end = h
        
        # نسخ جزء من الصورة الأصلية إلى الصورة الهدف مع إزاحة للأسفل
        if y_src_end > y_src_start and y_dst_end > y_dst_start:
            # حساب المنطقة المصدر والهدف
            src_region = img[y_src_start:y_src_end, :]
            
            # حساب الشفافية بناءً على رقم الإطار (الإطارات الأخيرة أكثر شفافية)
            alpha = 1.0 - (i / num_frames) * 0.8
            
            # تطبيق الطبقة على الصورة النهائية مع الشفافية
            if i == 0:
                # الطبقة الأولى تكون الصورة الأصلية
                result[y_dst_start:y_dst_end, :] = src_region
            else:
                # دمج الطبقات اللاحقة مع تأثير الشفافية
                dst_region = result[y_dst_start:y_dst_end, :]
                result[y_dst_start:y_dst_end, :] = cv2.addWeighted(
                    dst_region, 1.0, src_region, alpha, 0
                )
    
    # إضافة تأثير الحركة (motion blur)
    kernel_motion_blur = np.zeros((15, 15))
    kernel_motion_blur[7, :] = np.ones(15)
    kernel_motion_blur = kernel_motion_blur / 15
    result = cv2.filter2D(result, -1, kernel_motion_blur)
    
    return Image.fromarray(result)

def main():
    st.set_page_config(page_title="محرر تأثيرات الصور", layout="wide")
    st.title("🎭 محرر تأثيرات الصور")
    st.markdown("عزز صورك بتأثيرات مذهلة! قم بتحميل صورة أو صورة متحركة واختر تأثيرًا لتطبيقه.")
    
    uploaded_file = st.file_uploader("قم بتحميل صورة أو GIF", type=["png", "jpg", "jpeg", "gif"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        effect = st.selectbox("اختر تأثيرًا:", [
            "تأثير الخلل", "الضوضاء", "تأثير الشبح", "إزاحة RGB", "تمويه",
            "اكتشاف الحواف", "كرتون", "نيجاتيف", "سيبيا", "بارز", "تأثير السقوط للأمام"
        ])
        
        if effect == "تأثير الخلل":
            result = apply_glitch_effect(image)
        elif effect == "الضوضاء":
            result = apply_noise_effect(image)
        elif effect == "تأثير الشبح":
            result = apply_ghost_effect(image)
        elif effect == "إزاحة RGB":
            result = apply_rgb_shift(image)
        elif effect == "تمويه":
            result = apply_blur(image)
        elif effect == "اكتشاف الحواف":
            result = apply_edge_detection(image)
        elif effect == "كرتون":
            result = apply_cartoon_effect(image)
        elif effect == "نيجاتيف":
            result = apply_negative(image)
        elif effect == "سيبيا":
            result = apply_sepia(image)
        elif effect == "بارز":
            result = apply_emboss(image)
        elif effect == "تأثير السقوط للأمام":
            result = apply_tiktok_falling_effect(image)
        
        # استخدام المعلمة المحدثة
        st.image(result, caption=f"تم تطبيق {effect}", use_container_width=True)

if __name__ == "__main__":
    main()