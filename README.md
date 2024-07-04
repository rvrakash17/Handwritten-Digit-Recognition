# Handwritten Digit Recognition

## Introduction

This project involves creating a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset. The model is trained using TensorFlow and Keras and is deployed with a simple Tkinter-based graphical user interface (GUI) for digit prediction.

## Technologies Used

- **TensorFlow/Keras:** For building and training the neural network.
- **NumPy:** For numerical operations on data.
- **OpenCV:** For image processing.
- **Tkinter:** For creating the GUI.
- **PIL (Pillow):** For image handling in the GUI.
- **Matplotlib:** For visualizing sample images from the dataset.

## Model Architecture

1. **Convolutional Layers:**
   - `Conv2D` layer with 32 filters and 3x3 kernel.
   - `MaxPooling2D` layer with 2x2 pool size.
   - `Conv2D` layer with 64 filters and 3x3 kernel.
   - `MaxPooling2D` layer with 2x2 pool size.
   - `Conv2D` layer with 64 filters and 3x3 kernel.

2. **Fully Connected Layers:**
   - `Flatten` layer to reshape the data.
   - `Dense` layer with 64 units and ReLU activation.
   - `Dense` layer with 10 units (one for each digit) and softmax activation.

## Model Training

- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam
- **Metrics:** Accuracy
- **Epochs:** 5
- **Batch Size:** 64

## GUI Application

- **Drawing Canvas:** Allows users to draw digits.
- **Predict Button:** Predicts the drawn digit using the trained model.
- **Clear Button:** Clears the drawing canvas.
- **Save Button:** Optionally saves the drawn digit as an image file.

## Usage

1. **Setup and Dependencies:**
   Install the required packages using pip:
   ```sh
   pip install tensorflow keras numpy opencv-python pillow matplotlib
   ```

2. **Run the Training Script:**
   The training script is included and trains the CNN on the MNIST dataset. Once trained, the model is saved to `mnist.h5`.

3. **Run the GUI Application:**
   Execute the GUI script to use the trained model for digit prediction.

## Example Usage

1. **Training the Model:**
   ```python
   from tensorflow.keras import layers, models
   from keras.datasets import mnist
   from keras.utils import to_categorical
   
   (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
   
   model = models.Sequential([
       layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
       layers.MaxPooling2D((2, 2)),
       layers.Conv2D(64, (3, 3), activation='relu'),
       layers.MaxPooling2D((2, 2)),
       layers.Conv2D(64, (3, 3), activation='relu'),
       layers.Flatten(),
       layers.Dense(64, activation='relu'),
       layers.Dense(10, activation='softmax')
   ])
   
   model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
   
   train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
   test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
   train_labels = to_categorical(train_labels)
   test_labels = to_categorical(test_labels)
   
   model.fit(train_images, train_labels, epochs=5, batch_size=64)
   test_loss, test_acc = model.evaluate(test_images, test_labels)
   print(f"Test accuracy: {test_acc}")
   
   model.save('mnist.h5')
   ```

2. **Running the GUI Application:**
   ```python
   import tkinter as tk
   import numpy as np
   import cv2
   from PIL import Image, ImageDraw
   from keras.models import load_model
   
   model = load_model('mnist.h5')
   
   def event_function(event):
       x, y = event.x, event.y
       x1, y1 = x-30, y-30
       x2, y2 = x+30, y+30
       canvas.create_oval((x1, y1, x2, y2), fill='black')
       img_draw.ellipse((x1, y1, x2, y2), fill='white')
   
   def save():
       global count
       img_array = np.array(img)
       img_array = cv2.resize(img_array, (28, 28))
       cv2.imwrite(f'{count}.jpg', img_array)
       count += 1
   
   def clear():
       global img, img_draw
       canvas.delete('all')
       img = Image.new('RGB', (500, 500), (0, 0, 0))
       img_draw = ImageDraw.Draw(img)
       label_status.config(text='PREDICTED DIGIT: NONE')
   
   def predict():
       img_array = np.array(img)
       img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
       img_array = cv2.resize(img_array, (28, 28))
       img_array = img_array / 255.0
       img_array = img_array.reshape(1, 28, 28)
       result = model.predict(img_array)
       label = np.argmax(result, axis=1)
       label_status.config(text='PREDICTED DIGIT: ' + str(label))
   
   count = 0
   win = tk.Tk()
   
   canvas = tk.Canvas(win, width=500, height=500, bg='white')
   canvas.grid(row=0, column=0, columnspan=4)
   
   button_predict = tk.Button(win, text='PREDICT', bg='blue', fg='white', font='Helvetica 20 bold', command=predict)
   button_predict.grid(row=1, column=1)
   
   button_clear = tk.Button(win, text='CLEAR', bg='yellow', fg='white', font='Helvetica 20 bold', command=clear)
   button_clear.grid(row=1, column=2)
   
   label_status = tk.Label(win, text='PREDICTED DIGIT: NONE', bg='white', font='Helvetica 24 bold')
   label_status.grid(row=2, column=0, columnspan=4)
   
   canvas.bind('<B1-Motion>', event_function)
   img = Image.new('RGB', (500, 500), (0, 0, 0))
   img_draw = ImageDraw.Draw(img)
   
   win.mainloop()
   ```

## Visualization

- **Sample MNIST Images:**
   ```python
   import matplotlib.pyplot as plt
   from tensorflow.keras.datasets import mnist
   
   (X_train, Y_train), (_, _) = mnist.load_data()
   sample = 1
   image = X_train[sample]
   
   plt.figure()
   plt.imshow(image, cmap='gray')
   plt.title(f'Label: {Y_train[sample]}')
   plt.show()
   ```

## Author

- **Akash R**
