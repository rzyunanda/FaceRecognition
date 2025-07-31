import streamlit as st
import cv2
import os
import numpy as np
from datetime import datetime
from deepface import DeepFace
import pandas as pd

# Setup folder & log
FACE_FOLDER = "registered_faces"
LOG_FILE = "attendance_log.csv"
os.makedirs(FACE_FOLDER, exist_ok=True)

# Webcam capture
def capture_face():
    cap = cv2.VideoCapture(0)
    st.info("Tekan 's' untuk ambil gambar. Tekan 'q' untuk keluar.")
    img_captured = None

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Gagal akses kamera.")
            break
        cv2.imshow("Capture Wajah (Tekan 's')", frame)

        key = cv2.waitKey(1)
        if key == ord('s'):
            img_captured = frame
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return img_captured

# Save face
def save_face_image(image, name):
    path = os.path.join(FACE_FOLDER, f"{name}.jpg")
    cv2.imwrite(path, image)
    st.success(f"Wajah {name} berhasil disimpan.")

# Load registered faces
def load_registered_faces():
    faces = []
    for file in os.listdir(FACE_FOLDER):
        if file.endswith(".jpg"):
            img_path = os.path.join(FACE_FOLDER, file)
            faces.append((file.replace(".jpg", ""), cv2.imread(img_path)))
    return faces

# Recognize face
def recognize_face(image):
    registered = load_registered_faces()
    for name, reg_img in registered:
        try:
            result = DeepFace.verify(image, reg_img, enforce_detection=False)
            if result['verified']:
                return name
        except:
            continue
    return None

# Save attendance
def log_attendance(name):
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([[name, time]], columns=["Name", "Timestamp"])
    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)
    st.success(f"{name} tercatat hadir pada {time}.")

# UI
st.title("📸 Absensi Face Recognition dengan DeepFace")

option = st.sidebar.selectbox("Pilih Menu", ["Pendaftaran Wajah", "Verifikasi Kehadiran", "Lihat Log Absensi"])

if option == "Pendaftaran Wajah":
    st.header("📝 Daftarkan Wajah Baru")
    name = st.text_input("Masukkan Nama:")
    if st.button("Ambil Foto dan Simpan"):
        img = capture_face()
        if img is not None and name:
            save_face_image(img, name)

elif option == "Verifikasi Kehadiran":
    st.header("✅ Verifikasi Kehadiran")
    if st.button("Ambil Foto untuk Verifikasi"):
        img = capture_face()
        if img is not None:
            matched_name = recognize_face(img)
            if matched_name:
                log_attendance(matched_name)
            else:
                st.warning("Wajah tidak dikenali. Silakan daftar terlebih dahulu.")

elif option == "Lihat Log Absensi":
    st.header("📊 Log Kehadiran")
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        st.dataframe(df)
    else:
        st.info("Belum ada log absensi.")
