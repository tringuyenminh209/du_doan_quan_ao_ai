import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import fashion_mnist

# Bước 1: Tải dữ liệu Fashion MNIST
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Bước 2: Tiền xử lý
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Bước 3: Tạo mô hình CNN đơn giản
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 lớp ứng với 10 loại quần áo
])

# Bước 4: Compile mô hình
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Bước 5: Huấn luyện mô hình (chạy vài epoch là đủ để demo)
model.fit(x_train, y_train, epochs=5, validation_split=0.1, verbose=2)

# Bước 6: Lưu mô hình
model.save("model.h5")
print("✅ Mô hình đã được lưu thành công dưới tên model.h5")
