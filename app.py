from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Tải mô hình
try:
    model = load_model('model.h5')
    logging.info("Mô hình đã được tải thành công.")
except Exception as e:
    logging.error(f"Lỗi khi tải mô hình: {e}")
    model = None

# Dịch đa ngôn ngữ
translations = {
    "vi": {
        "title": "Phân loại sản phẩm bằng AI",
        "upload": "Tải ảnh lên",
        "predict": "Dự đoán",
        "result": "Kết quả"
    },
    "en": {
        "title": "AI Product Classifier",
        "upload": "Upload Image",
        "predict": "Predict",
        "result": "Result"
    },
    "jp": {
        "title": "AI製品分類",
        "upload": "画像をアップロード",
        "predict": "予測する",
        "result": "結果"
    }
}

LABELS_VI = [
    "Áo thun", "Quần dài", "Áo khoác", "Váy",
    "Áo khoác len", "Áo sơ mi", "Sandal", "Giày thể thao",
    "Túi xách", "Ủng"
]

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def process_image(img_path):
    """Xử lý ảnh và dự đoán nhãn."""
    try:
        img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
        img_array = image.img_to_array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        pred = model.predict(img_array)
        return LABELS_VI[np.argmax(pred)]
    except Exception as e:
        logging.error(f"Lỗi khi xử lý ảnh: {e}")
        return "Lỗi xử lý ảnh"

@app.route('/', methods=['GET', 'POST'])
def index():
    lang = request.args.get('lang', 'vi')
    label = None
    img_path = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)

            if model:
                label = process_image(img_path)
            else:
                label = "Mô hình không khả dụng."
        else:
            label = "Vui lòng tải lên một tệp hợp lệ."

    return render_template("index.html", label=label, img_path=img_path,
                           lang=lang, translations=translations)

if __name__ == '__main__':
    app.run(debug=True)