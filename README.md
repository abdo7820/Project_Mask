# 😷 Face Mask Detection - NTI AI Project

مشروع تطبيقي في مجال **رؤية الحاسوب** باستخدام تقنيات الذكاء الاصطناعي، يهدف إلى التمييز بين الأشخاص الذين **يرتدون الكمامات** والذين **لا يرتدونها** من خلال تحليل الصور، مما يخدم أهداف السلامة العامة خاصة في أوقات الأوبئة.

---

## 📝 الفكرة

نموذج ذكاء اصطناعي يعتمد على **الشبكات العصبية الالتفافية (CNN)** لتصنيف الصور إلى:

* ✅ **With Mask**
* ❌ **Without Mask**

---

## 🔄 Pipeline

### 🗂️ 1. البيانات

تم استخدام مجموعة بيانات مصنفة مسبقًا تحتوي على صور لأشخاص:

* **يرتدون الكمامة (With Mask)**
* **لا يرتدون الكمامة (Without Mask)**

> يمكنك استخدام إحدى المجموعات المعروفة مثل:
> [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)

---

### 🧹 2. المعالجة المسبقة

* **تغيير حجم الصور** إلى (128 × 128) بكسل لتوحيد الإدخال.
* **تحويل الصور إلى Tensors** باستخدام `img_to_array` و`expand_dims`.
* **تعزيز البيانات (Data Augmentation)** لتحسين التعميم باستخدام:

  * تدوير الصور (rotation)
  * قلب أفقي (horizontal flip)
  * تكبير/تصغير (zoom)

---

### 🧠 3. النمذجة

تم بناء نموذج باستخدام:

* **Convolutional Neural Network (CNN)** من خلال مكتبة `Keras`.

#### بنية النموذج (مثال):

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])
```

#### تقييم الأداء:

* **Accuracy**
* **Classification Report**
* **Confusion Matrix**

---

### 🖥️ 4. الواجهة التفاعلية (Gradio)

* يمكن للمستخدم رفع صورة من جهازه.
* يقوم النموذج بتحليل الصورة ويعرض النتيجة:

  * ✅ **With Mask**
  * ❌ **Without Mask**

---

## 📊 النتائج

* حقق النموذج دقة تصنيف تصل إلى **99%** على مجموعة الاختبار.
* أداء عالي بفضل استخدام **التحسين أثناء التدريب** و**Augmentation**.

---

## 🛠️ المتطلبات (Requirements)

```bash
pip install numpy tensorflow keras opencv-python gradio
```

---

## 🚀 طريقة التشغيل

```bash
python face_mask_app.py
```

---

## 📸 صور توضيحية

> أضف لقطات شاشة لواجهة المستخدم في Gradio، تُظهر مثالاً لصورة مع وبدون كمامة.

---

## 👨‍💻 عن المشروع

تم تطوير هذا المشروع كجزء من التدريب العملي في [NTI](https://www.nti.sci.eg/)، بهدف تعلم تطبيقات الرؤية الحاسوبية والذكاء الاصطناعي على مشكلات حقيقية تهم المجتمع.
