import cv2
import os

# Fungsi untuk membuat folder jika belum ada
def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Inisialisasi Webcam
cap = cv2.VideoCapture(0)

# Mendeteksi wajah dengan model haarcascade frontal face
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Folder untuk menyimpan gambar
save_dir = "captured_images"
make_dir(save_dir)

# Mengambil gambar dari webcam
ret, frame = cap.read()

# Jika berhasil mengambil gambar
if ret:
    # Deteksi wajah
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Menyimpan gambar original
    original_path = os.path.join(save_dir, "original.jpg")
    cv2.imwrite(original_path, frame)

    # Menyimpan gambar grayscale
    grayscale_path = os.path.join(save_dir, "grayscale.jpg")
    cv2.imwrite(grayscale_path, gray)

    # Mengubah ke black & white (thresholding)
    _, blackwhite = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    blackwhite_path = os.path.join(save_dir, "blackwhite.jpg")
    cv2.imwrite(blackwhite_path, blackwhite)

    # Crop wajah (hanya mengambil wajah yang pertama terdeteksi)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        cropped_face = frame[y:y+h, x:x+w]
        cropped_path = os.path.join(save_dir, "cropped_face.jpg")
        cv2.imwrite(cropped_path, cropped_face)
        print("Wajah terdeteksi dan di-crop.")
    else:
        print("Tidak ada wajah terdeteksi.")

    print("Gambar berhasil disimpan di folder:", save_dir)
else:
    print("Gagal mengambil gambar.")

# Melepaskan Webcam
cap.release()
cv2.destroyAllWindows()
