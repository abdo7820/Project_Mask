# 💼 NTI AI Projects: Heart Failure Prediction & Face Mask Detection

مشروعان تطبيقيان باستخدام الذكاء الاصطناعي ضمن تدريب NTI، يهدفان إلى حل مشكلتين واقعيتين في مجالي الرعاية الصحية والسلامة العامة:

---

## 🫀 1. Heart Failure Prediction

### 📝 الفكرة:

نموذج ذكي يتنبأ باحتمالية تعرض المريض لفشل في القلب بناءً على بياناته الطبية.

### 🔄 Pipeline:

* **البيانات:**
  تم استخدام مجموعة بيانات [Heart Failure Clinical Records Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) والتي تحتوي على معلومات طبية سريرية لـ 918 مريضًا.

* **المعالجة المسبقة:**

  * تنظيف القيم الشاذة والمفقودة (إن وجدت).
  * تطبيع البيانات باستخدام `StandardScaler`.
  * تقسيم البيانات إلى train/test.

* **النمذجة:**

  * تم استخدام نماذج متعددة:

    * S
    * Random Forest
    * XGBoost
  * تقييم الأداء:

    * Accuracy
    * Classification Report (Precision, Recall, F1-score)

* **الواجهة التفاعلية (Gradio):**

  * يمكن للمستخدم إدخال بيانات المريض (السمات الطبية).
  * يعرض النموذج النتيجة: **At Risk** أو **Not At Risk**.

### 📊 النتائج:

تم تحقيق دقة تصل إلى **87%** باستخدام نموذج **Random Forest** بعد الضبط والتحسين.

---

## 😷 2. Face Mask Detection

### 📝 الفكرة:

نموذج رؤية حاسوبية قادر على التمييز بين الأشخاص الذين يرتدون الكمامات والذين لا يرتدونها من الصور.

### 🔄 Pipeline:

* **البيانات:**
  استخدمت مجموعة بيانات "Face Mask Dataset" المصنفة إلى:

  * With Mask
  * Without Mask

* **المعالجة المسبقة:**

  * تغيير حجم الصور إلى (128x128).
  * تحويل الصور إلى مصفوفات رقمية (tensors).
  * استخدام **Data Augmentation** لتحسين التعميم.

* **النمذجة:**

  * تم بناء نموذج باستخدام:

    * Convolutional Neural Network (CNN)
  * تقييم الأداء:

    * Accuracy
    * Classification Report

* **الواجهة التفاعلية (Gradio):**

  * يمكن للمستخدم رفع صورة.
  * يعرض النموذج ما إذا كانت الصورة تحتوي على شخص **يرتدي كمامة** أو **لا يرتدي**.

### 📊 النتائج:

حقق النموذج دقة تصنيف تصل إلى **99%** باستخدام شبكة CNN.

---

## 🛠️ المتطلبات (Requirements)

```bash
pip install numpy pandas scikit-learn tensorflow keras matplotlib gradio opencv-python xgboost
```

---

## 🚀 طريقة التشغيل

### 1. تشغيل تطبيق التنبؤ بفشل القلب:

```bash
python heart_failure_app.py
```

### 2. تشغيل تطبيق اكتشاف الكمامات:

```bash
python face_mask_app.py
```

---

## 📸 صور توضيحية

يمكنك هنا إضافة لقطات شاشة من الواجهات الخاصة بـ Gradio.

---

## 👨‍💻 عن المشروع

هذا المشروع جزء من التدريب العملي المقدم من [NTI](https://www.nti.sci.eg/) ويهدف إلى تطبيق تقنيات الذكاء الاصطناعي لحل مشكلات واقعية باستخدام أدوات حديثة.

---

## 📫 التواصل

لأي استفسارات أو تعاون:

* 📧 [abdo.example@gmail.com](mailto:abdo.example@gmail.com)
* 💼 [LinkedIn](https://linkedin.com/in/your-profile)

---

هل تحب أن أجهز لك نسخة جاهزة مع استبدال `<XX%>` و `<اسم النموذج الأفضل>` بالقيم الحقيقية من نتائجك؟
