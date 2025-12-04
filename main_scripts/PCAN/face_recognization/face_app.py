


import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
import json
import os
import uuid, time
import cv2
import pyttsx3
import numpy as np
import threading
import hashlib
import subprocess
from deepface import DeepFace
from sklearn.cluster import KMeans
import mediapipe as mp

# ========================================
# CONFIG & FILES
# ========================================
DATA_FILE = "faces.json"
FACES_DIR = "faces"
os.makedirs(FACES_DIR, exist_ok=True)

CONFIG = {
    "similarity_threshold": 0.70,
    "fps_display": True,
    "font_scale": 0.8,
    "font_thickness": 2,
    "face_box_color": (0, 255, 0),
    "fps_text_color": (0, 255, 255),
    "unknown_text_color": (0, 0, 255),
    "camera_index": 0,
    "recognition_submit_interval": 0.8,
    "result_display_time": 3.0
}

# ========================================
# AUDIO UTIL
# ========================================
engine = pyttsx3.init()

def speak(text):
    """Speak text using pyttsx3."""
    try:
        # Use a separate thread to avoid blocking the main UI loop
        threading.Thread(target=lambda: (engine.say(text), engine.runAndWait())).start()
    except Exception as e:
        print(f"[Audio Error] {e}")


# ========================================
# JSON UTILITIES
# ========================================
def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                s = f.read().strip()
                if not s:
                    return {}
                return json.loads(s)
        except Exception as e:
            print(f"[load_data] Warning: failed to load {DATA_FILE}: {e}")
            return {}
    return {}

def save_data(data):
    out = {}
    for uid, info in data.items():
        emb = info.get("embedding")
        if isinstance(emb, np.ndarray):
            emb_to_save = emb.tolist()
        else:
            emb_to_save = emb
        out[uid] = {
            "name": info.get("name"),
            "embedding": emb_to_save,
            "image": info.get("image")
        }
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
#=========================================
# CHANGE USER NAME
#=========================================

def change_user_name():
    data = load_data()
    if not data:
        messagebox.showwarning("Warning", "No users exist. Please register first.")
        speak("No users exist. Please register first.")
        return

    win = tk.Toplevel(root)
    win.title("Change User Name")
    #speak("Change user name window opened")

    search_var = tk.StringVar()
    tk.Label(win, text="Search user:").pack()
    search_entry = tk.Entry(win, textvariable=search_var)
    search_entry.pack(fill=tk.X, padx=5)

    listbox = tk.Listbox(win, width=40, height=15)
    listbox.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 0))

    scrollbar = tk.Scrollbar(win, orient=tk.VERTICAL, command=listbox.yview)
    scrollbar.pack(side=tk.LEFT, fill=tk.Y)
    listbox.config(yscrollcommand=scrollbar.set)

    label_img = tk.Label(win, text="Select a user to rename")
    label_img.pack(pady=5)

    users_list = list(data.items())

    def update_listbox(filter_text=""):
        listbox.delete(0, tk.END)
        for uid, info in users_list:
            if filter_text.lower() in info["name"].lower() or filter_text.lower() in uid.lower():
                listbox.insert(tk.END, f"{uid}: {info['name']}")

    def on_search(event):
        update_listbox(search_var.get())

    def on_select(event):
        sel = listbox.curselection()
        if not sel:
            return
        uid = listbox.get(sel[0]).split(":")[0]
        user = data.get(uid, {})
        img_path = user.get("image")
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path).resize((150, 150))
            img_tk = ImageTk.PhotoImage(img)
            label_img.config(image=img_tk, text="")
            label_img.image = img_tk
        else:
            label_img.config(image="", text="No image available")

    def rename_selected():
        sel = listbox.curselection()
        if not sel:
            messagebox.showwarning("Warning", "No user selected.")
            #speak("No user selected for renaming")
            return
        uid = listbox.get(sel[0]).split(":")[0]
        old_name = data[uid]["name"]
        new_name = simpledialog.askstring("Rename", f"Enter new name for {old_name}:")
        if not new_name:
            speak("Rename cancelled")
            return
        data[uid]["name"] = new_name
        save_data(data)
        update_listbox(search_var.get())
        messagebox.showinfo("Renamed", f"Changed name from {old_name} to {new_name}")
        #speak(f"Changed name from {old_name} to {new_name}")

    update_listbox()
    search_entry.bind("<KeyRelease>", on_search)
    listbox.bind("<<ListboxSelect>>", on_select)

    btn_frame = tk.Frame(win)
    btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="Rename Selected", command=rename_selected).pack(side=tk.LEFT, padx=5)

# ========================================
# FACE CAPTURE
# ========================================
mp_face_mesh = mp.solutions.face_mesh

def get_head_pose(image, face_mesh):
    h, w, _ = image.shape
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0]
    landmark_indices = [1, 4, 33, 133, 362, 263, 61, 291, 199, 152, 105, 334, 50, 280]
    points_2d = np.array([[int(landmarks.landmark[idx].x * w),
                            int(landmarks.landmark[idx].y * h)] for idx in landmark_indices], dtype=np.float64)
    points_3d = np.array([[0.0, 0.0, 0.0],
                          [0.0, -10.0, -5.0],
                          [-30.0, -40.0, -30.0],
 [-10.0, -40.0, -30.0],
                          [10.0, -40.0, -30.0],
                          [30.0, -40.0, -30.0],
                          [-25.0, -70.0, -20.0],
                          [25.0, -70.0, -20.0],
                          [0.0, -100.0, -30.0],
                          [0.0, -120.0, -20.0],
                          [-20.0, -60.0, -20.0],
                          [20.0, -60.0, -20.0],
                          [0.0, -50.0, -20.0],
                          [0.0, -80.0, -20.0]])
    cam_matrix = np.array([[w, 0, w/2],
                           [0, w, h/2],
                           [0, 0, 1]])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    success, rot_vec, trans_vec = cv2.solvePnP(points_3d, points_2d, cam_matrix, dist_matrix)
    if not success:
        return None
    rot_mat, _ = cv2.Rodrigues(rot_vec)
    proj_mat = np.hstack((rot_mat, trans_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_mat)
    return euler_angles.flatten()

def capture_faces(angles=["front", "left", "right", "up", "down"], user_id="unknown"):
    cap = cv2.VideoCapture(CONFIG["camera_index"])
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")
    os.makedirs(FACES_DIR, exist_ok=True)
    embeddings = []
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
        for angle in angles:
            speak(f"Please look {angle}")
            print(f"Turn your face: {angle}")
            captured = False
            while not captured:
                ret, frame = cap.read()
                if not ret:
                    continue
                pose = get_head_pose(frame, face_mesh)
                if pose is not None:
                    pitch, yaw, roll = pose
                    cv2.putText(frame, f"Pitch: {pitch:.1f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame, f"Face: {angle.upper()}", (30, frame.shape[0]-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
                cv2.imshow("Face Capture - Press 'c' to capture", frame)
                if cv2.waitKey(1) & 0xFF == ord("c"):
                    try:
                        emb = DeepFace.represent(frame, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                        embeddings.append(emb)
                        captured = True
                        if angle == "front":
                            filename = os.path.join("faces", f"{user_id}.jpg")
                            cv2.imwrite(filename, frame)
                            print(f"Saved {filename}")
                    except Exception as e:
                        print("Face not detected:", e)
    cap.release()
    cv2.destroyAllWindows()
    if not embeddings:
        raise RuntimeError("No embeddings captured")
    if len(embeddings) > 1:
        kmeans = KMeans(n_clusters=1, random_state=0)
        kmeans.fit(embeddings)
        clustered = kmeans.cluster_centers_[0]
    else:
        clustered = embeddings[0]
    return clustered

# ========================================
# REGISTRATION UI
# ========================================
def register_user():
    data = load_data()
    name = simpledialog.askstring("Register", "Enter user name:")
    if not name:
        return

    cap = cv2.VideoCapture(CONFIG["camera_index"])
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open camera")
        speak("Error. Cannot open camera.")
        return

    #messagebox.showinfo("Face Check", "Look into the camera. Press 'c' to capture for duplicate face check.")
    speak("Look into the camera .")
    emb_np = None
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Face Check", frame)
        if cv2.waitKey(1) & 0xFF == ord("c"):
            try:
                emb = DeepFace.represent(frame, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                emb_np = np.array(emb, dtype=np.float32)
            except Exception:
                messagebox.showerror("Error", "Face not detected, try again.")
                speak("Face not detected. Please try again.")
            break
    cap.release()
    cv2.destroyAllWindows()

    if emb_np is None:
        return

    # Check if the person (name and face) is already in the database
    threshold = CONFIG.get("similarity_threshold", 0.70)
    for uid, info in data.items():
        db_emb = np.array(info.get("embedding"), dtype=np.float32)
        sim = float(np.dot(emb_np, db_emb) / (np.linalg.norm(emb_np) * np.linalg.norm(db_emb) + 1e-8))
        if sim >= threshold: # Face is a match
            if info['name'].lower() == name.lower(): # Check if name is also a match
                messagebox.showwarning( f"Registration blocked. '{name}' is already registered with this face.")
                speak(f"Registration cancel. This person is already registered.")
                return # Block registration
            else:
                messagebox.showwarning( f"Registration blocked. This face is already registered .")
                speak(f"Registration cancel. This face is already registered .")
                return # Block registration for a different name but same face


    temp_id = "tmp_" + str(int(time.time()))
    try:
        from_face = capture_faces(user_id=temp_id)
    except Exception as e:
        messagebox.showerror("Error", f"Face capture failed: {e}")
        speak("Face capture failed. Please try again.")
        return

    new_id = str(uuid.uuid4())[:10]
    tmp_img = os.path.join(FACES_DIR, f"{temp_id}.jpg")
    new_img_path = os.path.join(FACES_DIR, f"{new_id}.jpg")
    if os.path.exists(tmp_img):
        os.replace(tmp_img, new_img_path)

    data[new_id] = {
        "name": name,
        "embedding": np.array(from_face, dtype=np.float32).tolist(),
        "image": new_img_path
    }
    save_data(data)
    messagebox.showinfo("Success", f"Registered {name} (id: {new_id})")
    speak(f"Registration successful")
    
# ========================================
# SHOW USERS
# ========================================

def show_all_users():
    win = tk.Toplevel(root)
    win.title("All Users")

    search_var = tk.StringVar()
    tk.Label(win, text="Search user:").pack()
    search_entry = tk.Entry(win, textvariable=search_var)
    search_entry.pack(fill=tk.X, padx=5)

    listbox = tk.Listbox(win, width=40, height=15)
    listbox.pack(side=tk.LEFT, fill=tk.Y, padx=(5,0))

    scrollbar = tk.Scrollbar(win, orient=tk.VERTICAL, command=listbox.yview)
    scrollbar.pack(side=tk.LEFT, fill=tk.Y)
    listbox.config(yscrollcommand=scrollbar.set)

    label_img = tk.Label(win, text="Select user")
    label_img.pack(pady=5)

    users = load_data()
    users_list = list(users.items())

    def update_listbox(filter_text=""):
        listbox.delete(0, tk.END)
        for uid, info in users_list:
            if filter_text.lower() in info["name"].lower() or filter_text.lower() in uid.lower():
                listbox.insert(tk.END, f"{uid}: {info['name']}")

    update_listbox()

    def on_search(event):
        update_listbox(search_var.get())

    search_entry.bind("<KeyRelease>", on_search)

    def on_select(event):
        sel = listbox.curselection()
        if not sel:  
            return
        uid = listbox.get(sel[0]).split(":")[0]
        user = users.get(uid, {})
        img_path = user.get("image")
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path).resize((150,150))
            img_tk = ImageTk.PhotoImage(img)
            label_img.config(image=img_tk, text="")
            label_img.image = img_tk
        else:
            label_img.config(image="", text="No image available")

    listbox.bind("<<ListboxSelect>>", on_select)

# ========================================
# DELETE USER
# ========================================
def delete_user():
    data = load_data()
    if not data:
        messagebox.showwarning("Warning", "No users exist. Please register first.")
        return

    win = tk.Toplevel(root)
    win.title("Delete User")

    search_var = tk.StringVar()
    tk.Label(win, text="Search user:").pack()
    search_entry = tk.Entry(win, textvariable=search_var)
    search_entry.pack(fill=tk.X, padx=5)

    listbox = tk.Listbox(win, width=40, height=15, selectmode=tk.EXTENDED)
    listbox.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 0))

    scrollbar = tk.Scrollbar(win, orient=tk.VERTICAL, command=listbox.yview)   
    scrollbar.pack(side=tk.LEFT, fill=tk.Y)
    listbox.config(yscrollcommand=scrollbar.set)

    label_img = tk.Label(win, text="Select user(s) to delete")
    label_img.pack(pady=5)

    users_list = list(data.items())

    def update_listbox(filter_text=""):
        listbox.delete(0, tk.END)
        for uid, info in users_list:
            if filter_text.lower() in info["name"].lower() or filter_text.lower() in uid.lower():
                listbox.insert(tk.END, f"{uid}: {info['name']}")

    def on_search(event):
        update_listbox(search_var.get())

    def on_select(event):
        sel = listbox.curselection()
        if not sel:
            label_img.config(image="", text="Select user(s) to delete")
            return
        uid = listbox.get(sel[-1]).split(":")[0]
        user = data.get(uid, {})
        img_path = user.get("image")
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path).resize((150, 150))
            img_tk = ImageTk.PhotoImage(img)
            label_img.config(image=img_tk, text="")
            label_img.image = img_tk
        else:
            label_img.config(image="", text="No image available")

    def delete_selected():
        sel = listbox.curselection()
        if not sel:
            messagebox.showwarning("Warning", "No users selected.")
            return

        selected_users = [listbox.get(i) for i in sel]
        confirm = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete {len(selected_users)} user(s)?\n\n" + "\n".join(selected_users)
        )
        if not confirm:
            return

        for idx in sel[::-1]:
            uid = listbox.get(idx).split(":")[0]
            user = data.pop(uid, None)
            if user and user.get("image") and os.path.exists(user["image"]):
                os.remove(user["image"])

        save_data(data)
        update_listbox(search_var.get())
        label_img.config(image="", text="User(s) deleted")
        messagebox.showinfo("Deleted", f"Deleted {len(selected_users)} user(s).")

    def delete_all():
        confirm = messagebox.askyesno("Confirm Deletion", "Are you sure you want to delete ALL users?")
        if not confirm:
            return
        for uid, user in list(data.items()):
            if user.get("image") and os.path.exists(user["image"]):
                os.remove(user["image"])
        data.clear()
        save_data(data)
        update_listbox()
        label_img.config(image="", text="All users deleted")
        messagebox.showinfo("Deleted", "All users have been deleted.")

    update_listbox()
    search_entry.bind("<KeyRelease>", on_search)
    listbox.bind("<<ListboxSelect>>", on_select)

    btn_frame = tk.Frame(win)
    btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="Delete Selected", command=delete_selected).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text="Delete All", command=delete_all).pack(side=tk.LEFT, padx=5)

# ========================================
# RECOGNITION
# ========================================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def recognize_faces():
    data = load_data()
    cap = cv2.VideoCapture(CONFIG["camera_index"])
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open camera")
        speak("Error. Cannot open camera.")
        return

    welcomed = set()
    start_time = time.time()
    recognized = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face_crop = frame[y:y + h, x:x + w]
            try:
                emb = DeepFace.represent(face_crop, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                emb = np.array(emb)

                best, best_sim = "Unknown", -1
                for uid, info in data.items():
                    db_emb = np.array(info["embedding"])
                    sim = float(np.dot(emb, db_emb) / (np.linalg.norm(emb) * np.linalg.norm(db_emb) + 1e-8))
                    if sim > best_sim:
                        best_sim, best = sim, info["name"]

                if best_sim < CONFIG["similarity_threshold"]:
                    best = "Unknown"

                if best != "Unknown" and best not in welcomed:
                    welcomed.add(best)
                    recognized = True
                    # Voice alert for recognized face
                    speak(f"Welcome {best}.")

                    messagebox.showinfo("Welcome", f"Welcome {best}, starting DMS")
                    #speak(f"Welcome {best}, starting D M S")
                    cap.release()
                    cv2.destroyAllWindows()
                    subprocess.run(["python3", "Demo1_CAN_13_10_25.py"])   
                    return

            except Exception as e:
                print("Recognition error:", e)

        if not recognized and (time.time() - start_time) > 10:
            messagebox.showwarning("Unknown", "No recognized faces within time limit.")
            speak("No recognized faces within time limit.")
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
# ========================================
# QUIT APP
# ========================================

def quit_app():
    speak("Exiting application")
    root.quit()

# ========================================
# MAIN UI
# ========================================
root = tk.Tk()
root.title("Face Recognization")
root.geometry("300x300")

tk.Button(root, text="Register User", width=25, command=register_user).pack(pady=10)
tk.Button(root, text="Show Users", width=25, command=show_all_users).pack(pady=10)
tk.Button(root, text="Delete User", width=25, command=delete_user).pack(pady=10)
tk.Button(root, text="Recognize Faces", width=25, command=recognize_faces).pack(pady=10)
tk.Button(root, text="Change User Name", width=25, command=change_user_name).pack(pady=10)
tk.Button(root, text="Exit", width=25, command=quit_app).pack(pady=10)


root.mainloop()
