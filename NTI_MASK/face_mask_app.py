import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# تحميل النموذج
model = load_model("model.h5")

# أسماء الفئات
class_names = ['With Mask', 'Without Mask']
img_size = (128, 128)

# دالة التنبؤ
def predict_mask(image):
    # تأكد إن الصورة بصيغة RGB
    image = image.convert('RGB')
    
    # تغيير حجم الصورة
    image = image.resize(img_size)
    
    # تحويل الصورة إلى مصفوفة
    img_array = img_to_array(image)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    # توقع الفئة
    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    return f"{predicted_class} ({confidence:.2f}%)"

# إنشاء واجهة Gradio
interface = gr.Interface(
    fn=predict_mask,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Face Mask Detection",
    description="ارفع صورة وسيقوم النموذج بتحديد ما إذا كان الشخص يرتدي كمامة أو لا."
)

# تشغيل التطبيق
interface.launch()
