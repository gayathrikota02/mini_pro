from roboflow import Roboflow
import tkinter as tk
from tkinterdnd2 import TkinterDnD
from tkinter import filedialog, Label, Toplevel, StringVar, messagebox
from PIL import Image, ImageTk
from fpdf import FPDF


# Initialize Roboflow with YOLOv11 settings
rf = Roboflow(api_key="aSdKxesHWtPjM60DuRgU")
project = rf.workspace().project("brain-tumor-detection-pmivj")
model = project.version(7).model

import os

# Function to create sign-up window
def show_signup_window():
    signup_window = Toplevel(login_root)
    signup_window.title("Sign Up")
    signup_window.geometry("400x400")
    signup_window.configure(bg="lightblue")

    # Sign-up labels and entries
    signup_label = tk.Label(signup_window, text="Create an Account", font=("Arial", 20, "bold"), bg="lightblue")
    signup_label.pack(pady=20)

    username_label = tk.Label(signup_window, text="Username", font=("Comic Sans MS", 14), bg="lightblue")
    username_label.pack(pady=5)
    username_entry_signup = tk.Entry(signup_window, font=("Comic Sans MS", 14))
    username_entry_signup.pack(pady=5)

    password_label = tk.Label(signup_window, text="Password", font=("Comic Sans MS", 14), bg="lightblue")
    password_label.pack(pady=5)
    password_entry_signup = tk.Entry(signup_window, font=("Comic Sans MS", 14), show="*")
    password_entry_signup.pack(pady=5)

    # Function to save new user credentials
    def save_user():
        username = username_entry_signup.get()
        password = password_entry_signup.get()
        if username and password:
            # Save the username and password to a file (for simplicity)
            with open("users.txt", "a") as file:
                file.write(f"{username},{password}\n")
            messagebox.showinfo("Sign Up Success", "Account created successfully. You can now log in.")
            signup_window.destroy()  # Close sign-up window
        else:
            messagebox.showerror("Error", "Please fill in both fields.")

    # Sign up button
    signup_button = tk.Button(signup_window, text="Sign Up", command=save_user, font=("Comic Sans MS", 14), bg="green", fg="black")
    signup_button.pack(pady=20)


# Function to verify login
def verify_login():
    username = username_entry.get()
    password = password_entry.get()

    # Check credentials from the file (or database)
    if os.path.exists("users.txt"):
        with open("users.txt", "r") as file:
            users = file.readlines()
            for user in users:
                stored_username, stored_password = user.strip().split(",")
                if username == stored_username and password == stored_password:
                    start_main_app()  # Proceed to the main app after successful login
                    return
    messagebox.showerror("Login Failed", "Invalid username or password.")

# Function to start the main app
def start_main_app():
    login_root.destroy()  # Close the login window
    # Initialize the main app window here (You can continue with the rest of your app initialization)

# Initialize login window
login_root = TkinterDnD.Tk()
login_root.title("Login - Brain Tumor Detection")
login_root.geometry("600x600")  # Adjust window size to desired dimensions
login_root.configure(bg="lightblue")

# Background image for login page
background_image = Image.open("brain122.png")  # Change to your desired background image
background_image = background_image.resize((1600, 1000))  # Resize to full screen size of the window
background_photo = ImageTk.PhotoImage(background_image)

# Set the background image
background_label = tk.Label(login_root, image = background_photo)
background_label.place(relwidth=1, relheight=1)  # Makes the image cover the entire window

# Title on login page
title_label = tk.Label(login_root, text="Login to Brain Tumor Detection", font=("Arial", 35, "bold"), bg="black",fg="white")
title_label.pack(pady=20)
title_label.place(relx=0.5, rely=0.2, anchor="ne")

# Username and password entry fields
username_label = tk.Label(login_root, text="Username", font=("Comic Sans MS", 14), bg="black", fg="white")
username_label.pack(pady=10)
username_entry = tk.Entry(login_root, font=("Comic Sans MS", 14))
username_entry.pack(pady=5)

password_label = tk.Label(login_root, text="Password", font=("Comic Sans MS", 14), bg="black", fg="white")
password_label.pack(pady=10)
password_entry = tk.Entry(login_root, font=("Comic Sans MS", 14), show="*")
password_entry.pack(pady=5)

# Position username and password fields in the center
username_label.place(relx=0.3, rely=0.4, anchor="ne")
username_entry.place(relx=0.35, rely=0.45, anchor="ne")

password_label.place(relx=0.3, rely=0.5, anchor="ne")
password_entry.place(relx=0.35, rely=0.55, anchor="ne")

# Login button
login_button = tk.Button(login_root, text="Login", command=verify_login, font=("Comic Sans MS", 14), bg="black", fg="white")
login_button.pack(pady=20)
login_button.place(relx=0.24, rely=0.65, anchor="ne")

# Sign Up button
signup_button = tk.Button(login_root, text="Sign Up", command=show_signup_window, font=("Comic Sans MS", 14), bg="black", fg="white")
signup_button.pack(pady=10)
signup_button.place(relx=0.35, rely=0.65, anchor="ne")

login_root.mainloop()


# Initialize main app window
root = TkinterDnD.Tk()
root.title("Brain Tumor Detection with YOLOv11")
root.geometry("600x600")
root.configure(bg="lightblue")

# Set brain_tumor.jpg as the full background
background_image = Image.open("brain122.png")
background_image = background_image.resize((1600, 1000))
background_photo = ImageTk.PhotoImage(background_image)
background_label = Label(root, image=background_photo)
background_label.place(relwidth=1, relheight=1)

# Front page title
title_label = Label(root, text="Brain Tumor Detection", font=("Verdana", 35, "bold"), bg="black", fg="white")
title_label.pack(pady=20)
title_label.place(relx=0.5, rely=0.2, anchor="ne")


# Variable to hold prediction results
result_text = StringVar()


# Prediction and visualization with YOLOv11 (beta)
def predict_local_image(filepath):
    try:
        # Run prediction with YOLOv11 (beta) model
        prediction = model.predict(filepath, confidence=40, overlap=30).json()
        result_text.set("Prediction: " + str(prediction))

        # Save and display prediction image
        output_path = "predicted_image.jpg"
        model.predict(filepath, confidence=40, overlap=30).save(output_path)
        show_result_page(prediction, output_path)
    except Exception as e:
        messagebox.showerror("Prediction Error", f"Error during prediction: {e}")


# Show result page with updated features
# Show result page with updated features
def show_result_page(prediction, image_path):
    result_page = Toplevel(root)
    result_page.title("Prediction Results")
    result_page.geometry("600x700")
    result_page.configure(bg="lavender")

    # Show detected image
    img = Image.open(image_path).resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    img_label = Label(result_page, image=img_tk, bg="lavender")
    img_label.image = img_tk
    img_label.pack(pady=10)

    # Extract tumor name and confidence score
    tumor_prediction = prediction["predictions"][0]  # Assuming the first prediction is the most relevant
    tumor_name = tumor_prediction["class"]
    confidence_score = tumor_prediction["confidence"]  # Convert to percentage

    # Show detected tumor type and confidence score
    result_label = Label(result_page, text=f"Detected Tumor Type: {tumor_name}", font=("Arial", 16, "bold"),
                         bg="lavender", fg="darkviolet")
    result_label.pack(pady=5)


    # Show result details
    description_label = Label(result_page, text="Tumor Details:", font=("Arial", 16, "bold"), bg="lavender",
                              fg="purple")
    description_label.pack(pady=10)

    description_text = (
        "This tumor type is identified based on the YOLOv11 architecture, trained on specialized medical imaging data. "
        "Please consult a medical professional for further diagnosis and information."
    )
    description_info = Label(result_page, text=description_text, font=("Arial", 12), wraplength=500, bg="lavender",
                             fg="black")
    description_info.pack(pady=5)

    # Add Health Tips section
    health_tips_label = Label(result_page, text="Health Tips:", font=("Arial", 16, "bold"), bg="lavender", fg="green")
    health_tips_label.pack(pady=10)

    health_tips_text = (
        "1. Regular medical checkups can help detect tumors early.\n"
        "2. Consult a healthcare provider for personalized advice.\n"
        "3. Maintain a healthy lifestyle with proper nutrition and exercise."
    )
    health_tips_info = Label(result_page, text=health_tips_text, font=("Arial", 12), wraplength=500, bg="lavender",
                             fg="black")
    health_tips_info.pack(pady=5)

    # Generate report button
    generate_report_button = Button(result_page, text="Generate Report",
                                    command=lambda: generate_report(prediction),
                                    font=("Arial", 12), bg="lightgreen", fg="black")
    generate_report_button.pack(pady=20)



# Generate PDF report with tumor name, possible solution, and detected image (without prediction results)
def generate_report(prediction):
    tumor_name = prediction["predictions"][0]["class"]  # Tumor type
    solutions_text = (
        "1. Consult a doctor for a detailed diagnosis.\n"
        "2. Possible treatments include surgery, radiation therapy, or chemotherapy.\n"
        "3. Maintain a healthy lifestyle with a balanced diet and regular exercise."
    )

    report = FPDF()
    report.add_page()
    report.set_font("Arial", "B", 16)
    report.cell(200, 10, txt="Brain Tumor Detection Report", ln=True, align="C")

    report.set_font("Arial", "B", 12)
    report.ln(10)
    report.cell(200, 10, txt=f"Tumor Type: {tumor_name}", ln=True)


    report.set_font("Arial", size=12)
    report.ln(10)
    report.cell(200, 10, txt="Description: This tumor detection report is based on YOLOv11 analysis.", ln=True)

    report.ln(10)
    report.cell(200, 10, txt="Suggested Solutions:", ln=True)
    report.ln(5)
    report.multi_cell(0, 10, txt=solutions_text)

    # Add detected image to the report
    report.ln(10)
    image_path = "predicted_image.jpg"  # Assuming the detected image is saved here
    report.image(image_path, x=10, y=None, w=100)  # Adjust `x`, `y`, `w` as needed

    # Save the report
    report_path = "Tumor_Detection_Report.pdf"
    report.output(report_path)

    messagebox.showinfo("Report Generated", f"Report saved as {report_path}")


# Button to browse and select an image
def browse_file():
    filepath = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if filepath:
        predict_local_image(filepath)


import webbrowser
from tkinter import Button

# Add a button to display video about the project
def browse_file():
    filepath = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if filepath:
        predict_local_image(filepath)

# Function to open a video (local or online)
def open_video():
    video_path = "brain_tumor_detection.mp4"  # Path to the video file (local file or URL)
    # If it's a local file:
    # webbrowser.open(f"file:///{video_path}")  # Uncomment for local files

    # If you want to open an online video (e.g., YouTube link), use the URL:
    # video_url = "https://www.youtube.com/watch?v=your_video_id"
    # webbrowser.open(video_url)  # Uncomment for online video URL

    # Example of opening a local file
    webbrowser.open(video_path)  # For local video file


# Function to display project information
def show_project_info():
    info_window = Toplevel(root)
    info_window.title("Project Information")
    info_window.geometry("1600x1000")  # Adjust window size as needed
    info_window.configure(bg="lightyellow")

    # Add a title label for project information
    info_title = Label(info_window, text="Brain Tumor Detection Project", font=("Arial", 40, "bold"), bg="lightyellow",
                       fg="darkblue")
    info_title.pack(pady=20)

    # Add project description
    project_description = (
        "Title : Brain Tumor Detection.\n"
        "Domain	: Health Care.\n"
        "Guide : Dr. B. Saritha.\n"
        "Team Members : Gayathri (2451-22-748-007)\nSushmitha (2451-22-748-027)\nGrace (2451-22-748-305)\n"
        "Frontend Tools : Tkinter, Tkinter DnD2, PIL.\nBackend Tools : FPDF, YOLOv11 ."
        "\nDetecting brain tumors from MRI scans is crucial for early diagnosis and treatment. Traditional methods rely on manual inspection, which is time-consuming and prone to human error. This project aims to automate tumor detection by improving speed and accuracy.\n"
        "\nThe objective of this project is to develop an efficient brain tumor detection model using the YOLOv11 algorithm. By leveraging Roboflow for dataset preprocessing and augmentation, the model aims to accurately detect and localize brain tumors in medical images, providing rapid and reliable diagnostic support for clinicians.\n" 
        "\nThe project code is Python script which creates a GUI application for brain tumor detection using YOLOv11  with the help of Roboflow. The application allows users to upload MRI images for tumor detection, where the YOLOv11 model predicts the tumor type and displays the results.\n "
        "\nThese scans provide detailed images of brain structures, enabling the identification of irregularities that may indicate the presence of a tumor."
        "This project uses YOLOv11 (beta) architecture to detect brain tumors in medical images.\n"
        "\nKey Features:\n\n"
        "- YOLOv11 (beta) model for accurate object detection\n"
        "- User-friendly interface with easy navigation\n"
        "- Ability to generate a detailed report for diagnosis\n"
        "This project aims to assist doctors in early detection and provide timely treatment options."
    )

    # Create a label and place it to fill the entire window
    description_label = Label(info_window, text=project_description, font=("Comic Sans MS", 12), wraplength=1500,
                              bg="lightyellow", fg="black", justify="left", anchor="nw")
    description_label.pack(fill="both", expand=True, padx=0, pady=0)  # Pack the label to fill the window


# Function to add hover effect
def on_enter(e):
    e.widget['background'] = 'green'  # Change to your desired hover color

def on_leave(e):
    e.widget['background'] = e.widget.default_background  # Reset to default color

# File selection button
browse_button = Button(root, text="Select Image to Predict", command=browse_file, font=("Comic Sans MS", 14), bg="black",
                       fg="white")
browse_button.default_background = browse_button['background']  # Store default color
browse_button.bind("<Enter>", on_enter)
browse_button.bind("<Leave>", on_leave)
browse_button.pack(pady=20)
browse_button.place(relx=0.41, rely=0.4, anchor="ne")

# Button to open video (below the select image button)
video_button = Button(root, text="Watch Project Video", command=open_video, font=("Comic Sans MS", 14), bg="black", fg="white")
video_button.default_background = video_button['background']  # Store default color
video_button.bind("<Enter>", on_enter)
video_button.bind("<Leave>", on_leave)
video_button.pack(pady=10)
video_button.place(relx=0.4, rely=0.5, anchor="ne")

# Button to show project information
info_button = Button(root, text="Project Information", command=show_project_info, font=("Comic Sans MS", 14), bg="black",
                     fg="white")
info_button.default_background = info_button['background']  # Store default color
info_button.bind("<Enter>", on_enter)
info_button.bind("<Leave>", on_leave)
info_button.pack(pady=10)
info_button.place(relx=0.4, rely=0.6, anchor="ne")

# Run the GUI loop
root.mainloop()




