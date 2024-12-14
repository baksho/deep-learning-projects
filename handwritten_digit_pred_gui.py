import numpy as np
from tensorflow.keras.models import load_model
from tkinter import *
import tkinter as tk
from PIL import ImageGrab, Image

# Load the trained CNN model
model = load_model(
    r"C:\Users\Suvinava Basak\Documents\pythonScripts\deep_learning\digit_recog_mnist_cnn.keras"
)


# Function to preprocess the image and predict the digit
def predict_digit(img):
    # Resize the image to 28x28 while maintaining the aspect ratio (avoiding distortion)
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img = img.convert("L")  # Convert the image to grayscale
    img = np.array(img)

    # Thresholding to make the black pixels more distinct
    img = 255 - img  # Invert colors to make digits black
    img[img > 100] = 255  # Set pixels > 100 as white
    img[img <= 100] = 0  # Set pixels <= 100 as black

    # Display the image before passing it to the model
    # img_pil = Image.fromarray(img[0, :, :, 0])  # Convert to a PIL image for display
    # img_pil.show()  # Display the image

    # Reshape to match the input shape (1, 28, 28, 1) and normalize
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0

    # Predict the class
    prediction = model.predict(img)[0]
    digit = np.argmax(prediction)
    confidence = max(prediction)

    return digit, confidence


# GUI Application Class
class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0

        # Window title
        self.title("Handwritten Digit Recognition")

        # GUI Elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Draw a digit...", font=("Helvetica", 24))
        self.classify_btn = tk.Button(
            self,
            text="Recognize",
            command=self.classify_handwriting,
            font=("Helvetica", 14),
        )
        self.clear_btn = tk.Button(
            self, text="Clear", command=self.clear_canvas, font=("Helvetica", 14)
        )

        # Layout
        self.canvas.grid(row=0, column=0, pady=2, padx=2)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.clear_btn.grid(row=1, column=0, pady=2, padx=2)

        # Bind mouse events to draw on the canvas
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_canvas(self):
        """Clear the canvas."""
        self.canvas.delete("all")
        self.label.config(text="Draw a digit...")

    def classify_handwriting(self):
        """Capture the canvas, process the image, and classify the digit."""
        # Get the coordinates of the canvas
        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        # Capture the canvas content as an image
        img = ImageGrab.grab(bbox=(x, y, x1, y1))
        # Predict the digit
        digit, confidence = predict_digit(img)
        self.label.config(text=f"Digit: {digit}, Confidence: {int(confidence * 100)}%")

    def draw_lines(self, event):
        """Draw lines on the canvas."""
        self.x = event.x
        self.y = event.y
        r = 8  # Radius of the brush
        self.canvas.create_oval(
            self.x - r, self.y - r, self.x + r, self.y + r, fill="black"
        )


# Run the application
if __name__ == "__main__":
    app = App()
    app.mainloop()
