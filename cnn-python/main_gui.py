import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
from plate_recognition import extract_text_easyocr_smart


class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EasyOCR - Plate Recognition (Tkinter)")
        self.root.geometry("800x600")

        self.image_path = None

        # Label Gambar
        self.image_label = tk.Label(self.root, text="Tidak ada gambar", borderwidth=2, relief="groove")
        self.image_label.pack(pady=20)

        # Tombol pilih gambar
        btn_select = tk.Button(self.root, text="Pilih Gambar", command=self.select_image, width=20)
        btn_select.pack()

        # Tombol proses OCR
        btn_ocr = tk.Button(self.root, text="Proses OCR", command=self.run_ocr, width=20)
        btn_ocr.pack(pady=10)

        # Label hasil
        self.result_label = tk.Label(self.root, text="Hasil: -", font=("Arial", 16))
        self.result_label.pack(pady=20)

    # ------------------------------
    def select_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
        )

        if path:
            self.image_path = path

            # Load preview
            img = Image.open(path)
            img = img.resize((600, 350))
            img_tk = ImageTk.PhotoImage(img)

            self.image_label.configure(image=img_tk)
            self.image_label.image = img_tk
            self.result_label.config(text="Gambar siap diproses")

    # ------------------------------
    def run_ocr(self):
        if not self.image_path:
            messagebox.showwarning("Peringatan", "Pilih gambar terlebih dahulu!")
            return

        result = extract_text_easyocr_smart(self.image_path)
        self.result_label.config(text=f"Hasil: {result}")


# ------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()
