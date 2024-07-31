import os
import replicate
import base64
import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk, ImageSequence
from io import BytesIO
import requests
import json

# Set the Replicate API token directly in the script
os.environ['REPLICATE_API_TOKEN'] = 'r8_J7MOLyJC4b2ut3Jbku6N2SETapu9LUB24ARH6'
API_KEY = os.getenv('REPLICATE_API_TOKEN')

def show_page_1():
    clear_page()
    label = tk.Label(root, text="Welcome to Age AI", font=("Helvetica", 98, "bold"), bg="#375D77", fg="white")
    label.pack(pady=250)
    button = tk.Button(root, text="Click To Start!", font=("Helvetica", 24, "bold"), bg="#375D77", fg="white", borderwidth=0, highlightthickness=0, relief='flat', command=show_page_2)
    button.pack(pady=50)

def show_page_2():
    clear_page()
    frame = tk.Frame(root, bg='#000000')
    frame.pack(padx=20, pady=20, fill='both', expand=True)
    label = tk.Label(frame, text="Position your face in the center of the camera", font=("Helvetica", 14, "bold"), bg='#000000', fg='white')
    label.pack(pady=10)
    video_label = tk.Label(frame)
    video_label.pack()
    capture_button = tk.Button(frame, text="Capture Image", font=("Helvetica", 14), bg="#4CAF50", fg="white", command=lambda: capture_image(video_label, cap))
    capture_button.pack(pady=20)
    global cap
    cap = open_camera()
    if not cap:
        label = tk.Label(frame, text="Unable to access the camera", bg='#000000', fg='red', font=("Helvetica", 14, "bold"))
        label.pack(pady=20)
    else:
        show_frame(video_label, cap)

def clear_page():
    for widget in root.winfo_children():
        widget.destroy()

def open_camera():
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap
        cap.release()
    return None

def show_frame(label, cap):
    ret, frame = cap.read()
    if ret:
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
    label.after(10, show_frame, label, cap)

def capture_image(label, cap):
    try:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('captured_image.jpg', frame)
            label.imgtk = None
            cap.release()
            print("Image captured successfully.")
            process_image('captured_image.jpg')
        else:
            print("Failed to capture image.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while capturing the image: {str(e)}")
        print("Capture error:", e)

def process_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        print("Image encoded successfully.")

        model = replicate.models.get("yuval-alaluf/sam")
        version = model.versions.get("9222a21c181b707209ef12b5e0d7e94c994b58f01c7b2fec075d2e892362f13c")
        print("Model and version fetched successfully.")

        prediction = replicate.run(
            version,
            input={"image": f"data:image/jpeg;base64,{encoded_image}", "target_age": "default"}
        )
        
        print("Prediction response:", prediction)  # Debug statement
        print("Type of prediction response:", type(prediction))  # Debug statement

        # Check if the prediction is a string (error message)
        if isinstance(prediction, str):
            messagebox.showerror("Error", f"Failed to process the image: {prediction}")
            return

        # If prediction is a bytes or bytearray, decode it
        if isinstance(prediction, (bytes, bytearray)):
            prediction = prediction.decode('utf-8')
        
        # If prediction is still a string, try to parse it as JSON
        if isinstance(prediction, str):
            prediction = json.loads(prediction)
        
        # Ensure we get the correct URL from the response
        output_url = prediction.get('output')
        if not output_url:
            messagebox.showerror("Error", "No output URL found in the response")
            return

        print("Output URL:", output_url)  # Debug statement
        display_aged_image(output_url)
    except json.JSONDecodeError as e:
        messagebox.showerror("Error", f"JSON decode error: {str(e)}")
        print("JSON decode error:", e)
    except KeyError as e:
        messagebox.showerror("Error", f"Key error: {str(e)}")
        print("Key error:", e)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        print("Error:", e)

def display_aged_image(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img_data = response.content

        # Load the GIF image
        img = Image.open(BytesIO(img_data))

        clear_page()

        # Create a label to display the GIF
        label = tk.Label(root)
        label.pack()

        # Function to display each frame of the GIF
        def update_frame(ind):
            frame = ImageTk.PhotoImage(img.seek(ind))
            label.config(image=frame)
            label.image = frame
            ind = (ind + 1) % img.n_frames
            root.after(100, update_frame, ind)

        # Start displaying the GIF
        update_frame(0)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to display the image: {str(e)}")
        print("Error:", e)

# Create the main window
root = tk.Tk()
root.title("Age AI")
root.config(bg='black')
root.attributes('-fullscreen', True)

# Show the first page
show_page_1()

# Start the main loop
root.mainloop()
