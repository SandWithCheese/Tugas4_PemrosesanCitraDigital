import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import os
from plate_recognition import extract_text_easyocr_smart
from infer_buah import infer

def create_gradient(canvas, width, height, color1, color2):
    r1, g1, b1 = canvas.winfo_rgb(color1)
    r2, g2, b2 = canvas.winfo_rgb(color2)
    r_ratio = (r2 - r1) / height
    g_ratio = (g2 - g1) / height
    b_ratio = (b2 - b1) / height
    for i in range(height):
        nr = int(r1 + (r_ratio * i))
        ng = int(g1 + (g_ratio * i))
        nb = int(b1 + (b_ratio * i))
        color = f"#{nr//256:02x}{ng//256:02x}{nb//256:02x}"
        canvas.create_line(0, i, width, i, fill=color)

def create_rounded_rectangle(canvas, x1, y1, x2, y2, radius=25, **kwargs):
    points = [
        x1+radius, y1, x1+radius, y1,
        x2-radius, y1, x2-radius, y1,
        x2, y1, x2, y1+radius,
        x2, y1+radius, x2, y2-radius,
        x2, y2-radius, x2, y2,
        x2-radius, y2, x2-radius, y2,
        x1+radius, y2, x1+radius, y2,
        x1, y2, x1, y2-radius,
        x1, y2-radius, x1, y1+radius,
        x1, y1+radius, x1, y1
    ]
    return canvas.create_polygon(points, **kwargs, smooth=True)

class ModernButton(tk.Canvas):
    def __init__(self, parent, text, command=None, width=200, height=40,
                 bg_color="#667EEA", hover_color="#5568D3", text_color="white", **kwargs):
        super().__init__(parent, width=width, height=height,
                        highlightthickness=0, bg=parent.cget('bg'), **kwargs)
        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.width = width
        self.height = height
        self.shadow_rect = create_rounded_rectangle(
            self, 3, 3, width-1, height+2, radius=12,
            fill="#1a1a2e", outline=""
        )
        self.button_rect = create_rounded_rectangle(
            self, 2, 2, width-2, height-2, radius=12,
            fill=bg_color, outline=""
        )
        self.text_item = self.create_text(
            width/2, height/2, text=text,
            fill=text_color, font=("Segoe UI", 11, "bold")
        )
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)

    def on_enter(self, event):
        self.itemconfig(self.button_rect, fill=self.hover_color)
        self.config(cursor="hand2")

    def on_leave(self, event):
        self.itemconfig(self.button_rect, fill=self.bg_color)
        self.config(cursor="")

    def on_click(self, event):
        if self.command:
            self.command()

class MultiAppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CNN Vision Hub - Fruit Segmentation & Plate Recognition")
        self.root.geometry("1200x700")
        self.root.resizable(False, False)

        self.bg_canvas = tk.Canvas(self.root, width=1200, height=700, highlightthickness=0)
        self.bg_canvas.place(x=0, y=0)
        create_gradient(self.bg_canvas, 1200, 700, "#667EEA", "#764BA2")

        self.image_path = None
        self.current_mode = None
        self.setup_ui()

    def setup_ui(self):
        header_frame = tk.Frame(self.root, bg="#2C2E4A", height=80)
        header_frame.place(x=0, y=0, width=1200, height=80)

        tk.Label(
            header_frame,
            text="🎯 CNN Vision Hub",
            font=("Segoe UI", 28, "bold"),
            bg="#2C2E4A",
            fg="#FFFFFF"
        ).place(x=30, y=15)
        sidebar = tk.Frame(self.root, bg="#2C3E50", width=320)
        sidebar.place(x=0, y=80, width=320, height=620)

        # Mode Selection Section
        tk.Label(
            sidebar,
            text="📋 Processing Mode",
            font=("Segoe UI", 12, "bold"),
            bg="#2C3E50",
            fg="#FFFFFF"
        ).place(x=20, y=20)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Modern.TCombobox',
                       fieldbackground='#34495E',
                       background='#34495E',
                       foreground='white',
                       arrowcolor='#667EEA',
                       borderwidth=1,
                       relief='flat',
                       padding=5)

        self.mode_var = tk.StringVar(value="Pilih Mode")
        self.mode_dropdown = ttk.Combobox(
            sidebar,
            textvariable=self.mode_var,
            values=["OCR Plat Nomor", "Segmentasi Buah"],
            state="readonly",
            font=("Segoe UI", 10),
            style='Modern.TCombobox',
            width=30
        )
        self.mode_dropdown.place(x=20, y=55)
        self.mode_dropdown.bind("<<ComboboxSelected>>", self.on_mode_change)

        # Actions Section
        tk.Label(
            sidebar,
            text="⚙️ Actions",
            font=("Segoe UI", 12, "bold"),
            bg="#2C3E50",
            fg="#FFFFFF"
        ).place(x=20, y=110)

        self.btn_select = ModernButton(
            sidebar,
            text="📁 Pilih Gambar",
            command=self.select_image,
            width=280,
            height=42,
            bg_color="#667EEA",
            hover_color="#5568D3"
        )
        self.btn_select.place(x=20, y=150)

        self.btn_process = ModernButton(
            sidebar,
            text="▶️ Proses",
            command=self.process,
            width=280,
            height=42,
            bg_color="#E74C3C",
            hover_color="#C0392B"
        )
        self.btn_process.place(x=20, y=205)

        # Status Panel
        tk.Label(
            sidebar,
            text="📊 Status & Results",
            font=("Segoe UI", 12, "bold"),
            bg="#2C3E50",
            fg="#FFFFFF"
        ).place(x=20, y=270)

        status_panel = tk.Frame(sidebar, bg="#34495E", relief=tk.RAISED, bd=1)
        status_panel.place(x=15, y=310, width=290, height=280)

        tk.Label(
            status_panel,
            text="Mode:",
            font=("Segoe UI", 9, "bold"),
            bg="#34495E",
            fg="#BDC3C7"
        ).place(x=12, y=12)

        self.mode_label = tk.Label(
            status_panel,
            text="Belum dipilih",
            font=("Segoe UI", 10),
            bg="#34495E",
            fg="#667EEA",
            wraplength=260,
            anchor="w",
            justify="left"
        )
        self.mode_label.place(x=12, y=32)

        tk.Label(
            status_panel,
            text="Status:",
            font=("Segoe UI", 9, "bold"),
            bg="#34495E",
            fg="#BDC3C7"
        ).place(x=12, y=70)

        self.status_label = tk.Label(
            status_panel,
            text="Menunggu input",
            font=("Segoe UI", 10),
            bg="#34495E",
            fg="#F39C12",
            wraplength=260,
            anchor="w",
            justify="left"
        )
        self.status_label.place(x=12, y=90)

        tk.Label(
            status_panel,
            text="Hasil:",
            font=("Segoe UI", 9, "bold"),
            bg="#34495E",
            fg="#BDC3C7"
        ).place(x=12, y=130)

        self.result_label = tk.Label(
            status_panel,
            text="-",
            font=("Segoe UI", 14, "bold"),
            bg="#34495E",
            fg="#2ECC71",
            wraplength=260,
            justify="left",
            anchor="nw"
        )
        self.result_label.place(x=12, y=150, width=266, height=110)

        # Main Content Frame
        self.content_frame = tk.Frame(self.root, bg="#1A1A1A")
        self.content_frame.place(x=335, y=95, width=845, height=605)

        tk.Label(
            self.content_frame,
            text="🖼️ Image Preview",
            font=("Segoe UI", 13, "bold"),
            bg="#1A1A1A",
            fg="#FFFFFF"
        ).place(x=15, y=10)

        self.image_canvas = tk.Canvas(
            self.content_frame,
            width=815,
            height=470,
            bg="#2C2C2C",
            highlightthickness=1,
            highlightbackground="#444444"
        )
        self.image_canvas.place(x=15, y=40)

        self.preview_result_label = tk.Label(
            self.content_frame,
            text="",
            font=("Segoe UI",40, "bold"),
            bg="#1A1A1A",
            fg="#2ECC71",
            wraplength=815,
            justify="center"
        )

        self.placeholder_text = self.image_canvas.create_text(
            407, 235,
            text="Belum ada gambar\n\nSilakan pilih gambar untuk memulai",
            font=("Segoe UI", 14),
            fill="#666666",
            justify="center"
        )

    def on_mode_change(self, event=None):
        mode = self.mode_var.get()
        if mode == "OCR Plat Nomor":
            self.current_mode = "plate"
            self.mode_label.config(text="OCR Plat Nomor", fg="#667EEA")
            self.result_label.config(text="-")
            self.preview_result_label.config(text="")
        elif mode == "Segmentasi Buah":
            self.current_mode = "fruit"
            self.mode_label.config(text="Segmentasi Buah", fg="#667EEA")
            self.result_label.config(text="-")
            self.preview_result_label.config(text="")
        self.status_label.config(text="Mode aktif", fg="#2ECC71")

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Pilih Gambar",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if path:
            self.image_path = path
            img = Image.open(path)
            img.thumbnail((815, 470), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            self.image_canvas.delete("all")
            self.image_canvas.create_image(407, 235, image=img_tk)
            self.image_canvas.image = img_tk
            self.status_label.config(text="Gambar berhasil dimuat", fg="#2ECC71")

    def process(self):
        if not self.image_path:
            messagebox.showwarning("Peringatan", "Pilih gambar terlebih dahulu!")
            return
        if not self.current_mode:
            messagebox.showwarning("Peringatan", "Pilih mode terlebih dahulu!")
            return
        self.status_label.config(text="Memproses...", fg="#F39C12")
        self.root.update()

        if self.current_mode == "plate":
            self.run_plate_ocr()
        else:
            self.run_fruit_seg()

    def run_plate_ocr(self):
        try:
            result = extract_text_easyocr_smart(self.image_path)
            self.result_label.config(text=result, fg="#2ECC71")
            self.preview_result_label.config(text="")
            self.status_label.config(text="Proses selesai", fg="#2ECC71")
        except Exception as e:
            self.result_label.config(text=f"Error: {str(e)}", fg="#E74C3C")
            self.preview_result_label.config(text="")
            self.status_label.config(text="Gagal", fg="#E74C3C")

    def run_fruit_seg(self):
        try:
            CLASSES = [
                "Fruits","Apple","Banana","Grapes","Kiwi",
                "Mango","Orange","Pineapple","Strawberry","Sugarapple","Watermelon"
            ]
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.normpath(os.path.join(script_dir, "unet_multitask_best.pth"))
            detected, overlay_path = infer(
                image_path=self.image_path,
                model_path=model_path,
                class_names=CLASSES
            )
            if len(detected) == 0:
                self.result_label.config(text="Tidak ada buah", fg="#F39C12")
            else:
                fruits_str = ", ".join(detected)
                self.result_label.config(text=f"{fruits_str}", fg="#2ECC71")
            if os.path.exists(overlay_path):
                img = Image.open(overlay_path)
                img.thumbnail((815, 470), Image.Resampling.LANCZOS)
                img_tk = ImageTk.PhotoImage(img)
                self.image_canvas.delete("all")
                self.image_canvas.create_image(407, 235, image=img_tk)
                self.image_canvas.image = img_tk
            self.preview_result_label.config(text="")
            self.status_label.config(text="Selesai", fg="#2ECC71")
        except Exception as e:
            self.result_label.config(text=f"Error: {str(e)}", fg="#E74C3C")
            self.status_label.config(text="Gagal", fg="#E74C3C")

if __name__ == "__main__":
    root = tk.Tk()
    app = MultiAppGUI(root)
    root.mainloop()
