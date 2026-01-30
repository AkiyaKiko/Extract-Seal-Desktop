# app_tk.py
from __future__ import annotations
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageGrab

from seal_core import extract_seals, Circle


def cv_bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def cv_bgra_to_pil(img_bgra: np.ndarray) -> Image.Image:
    rgba = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGBA)
    return Image.fromarray(rgba)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ExtractSeal (Python/Tkinter)")
        self.geometry("1120x680")

        self.img_path: str | None = None
        self.img_bgr: np.ndarray | None = None
        self.result = None
        self.preview_index = None

        self._build_ui()
        self.bind_all("<Control-v>", self._on_paste_hotkey)
        self.bind_all("<Control-V>", self._on_paste_hotkey)

    def _build_ui(self):
        # Top controls
        frm = tk.Frame(self)
        frm.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        tk.Button(frm, text="选择图片", command=self.choose_image).pack(side=tk.LEFT)
        tk.Button(frm, text="从剪贴板粘贴(Ctrl+V)", command=self.paste_image).pack(side=tk.LEFT, padx=(8, 0))

        self.path_var = tk.StringVar(value="")
        tk.Entry(frm, textvariable=self.path_var, width=60).pack(side=tk.LEFT, padx=8)

        tk.Label(frm, text="颜色(#RRGGBB)").pack(side=tk.LEFT, padx=(10, 4))
        self.color_var = tk.StringVar(value="#ff0000")
        tk.Entry(frm, textvariable=self.color_var, width=10).pack(side=tk.LEFT)

        tk.Label(frm, text="Hue±").pack(side=tk.LEFT, padx=(10, 4))
        self.hue_var = tk.IntVar(value=10)
        tk.Scale(frm, from_=1, to=30, orient=tk.HORIZONTAL, variable=self.hue_var, length=140).pack(side=tk.LEFT)

        tk.Label(frm, text="Smin").pack(side=tk.LEFT, padx=(10, 4))
        self.smin_var = tk.IntVar(value=50)
        tk.Scale(frm, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.smin_var, length=140).pack(side=tk.LEFT)

        tk.Label(frm, text="Vmin").pack(side=tk.LEFT, padx=(10, 4))
        self.vmin_var = tk.IntVar(value=50)
        tk.Scale(frm, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.vmin_var, length=140).pack(side=tk.LEFT)

        frm2 = tk.Frame(self)
        frm2.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        tk.Label(frm2, text="裁剪放大").pack(side=tk.LEFT)
        self.scale_var = tk.DoubleVar(value=1.2)
        tk.Scale(frm2, from_=1.0, to=2.0, resolution=0.05, orient=tk.HORIZONTAL, variable=self.scale_var, length=160).pack(side=tk.LEFT, padx=6)

        tk.Label(frm2, text="最多输出").pack(side=tk.LEFT, padx=(10, 4))
        self.max_var = tk.IntVar(value=6)
        tk.Spinbox(frm2, from_=1, to=12, textvariable=self.max_var, width=4).pack(side=tk.LEFT)

        tk.Button(frm2, text="提取", command=self.run_extract).pack(side=tk.LEFT, padx=10)
        self.save_btn = tk.Button(frm2, text="保存选中", command=self.save_selected, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT)


        self.info_var = tk.StringVar(value="")
        tk.Label(frm2, textvariable=self.info_var, fg="#444").pack(side=tk.LEFT, padx=12)

        # Main panels
        main = tk.Frame(self)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left: source + processed preview
        left = tk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(left, text="原图").pack(anchor="w")
        self.src_label = tk.Label(left, bd=1, relief=tk.SOLID, width=420, height=260)
        self.src_label.pack(fill=tk.BOTH, expand=False, pady=(0, 10))

        tk.Label(left, text="处理后（纯色+透明）").pack(anchor="w")
        self.stamp_label = tk.Label(left, bd=1, relief=tk.SOLID, width=420, height=260)
        self.stamp_label.pack(fill=tk.BOTH, expand=False)

        # Right: list + crop preview
        right = tk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0))

        tk.Label(right, text="检测结果（点击预览）").pack(anchor="w")
        self.listbox = tk.Listbox(right, height=12)
        self.listbox.pack(fill=tk.X, expand=False)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)

        tk.Label(right, text="裁剪预览").pack(anchor="w", pady=(10, 0))
        self.crop_label = tk.Label(right, bd=1, relief=tk.SOLID, width=420, height=260)
        self.crop_label.pack(fill=tk.BOTH, expand=False)

        # Hold references to PhotoImage to prevent GC
        self._src_photo = None
        self._stamp_photo = None
        self._crop_photo = None

    def _on_paste_hotkey(self, _event):
        self.paste_image()

    def paste_image(self):
        """
        从剪贴板读取图像：
        - Windows: ImageGrab.grabclipboard() 通常可直接拿到 PIL.Image
        - 如果剪贴板是文件路径列表（例如复制了文件），也兼容读第一张
        """
        try:
            data = ImageGrab.grabclipboard()
        except Exception as e:
            messagebox.showerror("错误", f"读取剪贴板失败：{e}")
            return

        if data is None:
            messagebox.showwarning("提示", "剪贴板里没有图片。请先截图并复制（或使用截图工具的“复制”）。")
            return

        pil_img = None

        # 情况1：直接是 PIL.Image
        if isinstance(data, Image.Image):
            pil_img = data

        # 情况2：剪贴板里是文件路径列表（你复制了图片文件）
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
            p = data[0]
            if os.path.exists(p):
                try:
                    pil_img = Image.open(p).convert("RGB")
                    self.img_path = p
                    self.path_var.set(p)
                except Exception as e:
                    messagebox.showerror("错误", f"打开剪贴板文件失败：{e}")
                    return
            else:
                messagebox.showwarning("提示", "剪贴板里是路径，但文件不存在。")
                return

        else:
            messagebox.showwarning("提示", "剪贴板内容不是图片（也不是图片文件路径）。")
            return

        # 统一转为 RGB（有些截图会是 RGBA）
        pil_img = pil_img.convert("RGB")

        # PIL -> OpenCV BGR
        np_img = np.array(pil_img)               # RGB
        img_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

        self.img_bgr = img_bgr
        # 如果是纯截图，没有文件路径，就标记为 clipboard
        if not self.img_path:
            self.img_path = "clipboard.png"
            self.path_var.set("(剪贴板图像)")

        # 刷新 UI 预览
        self._show_pil(self.src_label, cv_bgr_to_pil(img_bgr), kind="src")
        self.info_var.set("已从剪贴板粘贴图像，点击“提取”开始处理")
        self._reset_results()


    def choose_image(self):
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.webp")]
        )
        if not path:
            return
        self.img_path = path
        self.path_var.set(path)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("错误", "读取图片失败：格式不支持或路径错误")
            return
        self.img_bgr = img
        self._show_pil(self.src_label, cv_bgr_to_pil(img), kind="src")
        self.info_var.set("已加载图片，点击“提取”开始处理")
        self._reset_results()

    def _reset_results(self):
        self.result = None
        self.listbox.delete(0, tk.END)
        self.crop_label.configure(image="")
        self._crop_photo = None
        self.save_btn.configure(state=tk.DISABLED)

    def run_extract(self):
        if self.img_bgr is None:
            messagebox.showwarning("提示", "请先选择图片")
            return

        try:
            res = extract_seals(
                self.img_bgr,
                color_hex=self.color_var.get().strip(),
                hue_range=int(self.hue_var.get()),
                s_min=int(self.smin_var.get()),
                v_min=int(self.vmin_var.get()),
                scale_factor=float(self.scale_var.get()),
                max_count=int(self.max_var.get()),
            )
        except Exception as e:
            messagebox.showerror("错误", str(e))
            return

        self.result = res

        # show processed stamp
        stamp_pil = cv_bgra_to_pil(res["stamp_bgra"])
        self._show_pil(self.stamp_label, stamp_pil, kind="stamp")

        # fill listbox
        self.listbox.delete(0, tk.END)
        circles: list[Circle] = res["circles"]
        for i, c in enumerate(circles):
            self.listbox.insert(tk.END, f"#{i+1}  x={c.x:.1f}  y={c.y:.1f}  r={c.radius:.1f}")

        if len(res["crops"]) > 0:
            self.listbox.selection_set(0)
            self.listbox.event_generate("<<ListboxSelect>>")

        self.info_var.set(f"检测到 {len(res['crops'])} 个圆形印章")
        self.save_btn.configure(state=tk.DISABLED)

    def on_select(self, _event):
        if not self.result:
            return
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = int(sel[0])
        crops: list[np.ndarray] = self.result["crops"]
        if idx < 0 or idx >= len(crops):
            return
        crop_pil = cv_bgra_to_pil(crops[idx])
        self._show_pil(self.crop_label, crop_pil, kind="crop")
        self.save_btn.configure(state=tk.NORMAL)

    def save_all(self):
        if not self.result or not self.img_path:
            return
        crops: list[np.ndarray] = self.result["crops"]
        if not crops:
            return

        folder = filedialog.askdirectory(title="选择保存目录")
        if not folder:
            return

        base = os.path.splitext(os.path.basename(self.img_path))[0]
        saved = 0
        for i, crop in enumerate(crops):
            out_path = os.path.join(folder, f"{base}_stamp_{i+1}.png")
            # crop is BGRA, cv2.imwrite can write png with alpha
            cv2.imwrite(out_path, crop)
            saved += 1

        messagebox.showinfo("完成", f"已保存 {saved} 张 PNG 到：\n{folder}")

    def save_selected(self):
        if not self.result or not self.img_path:
            return

        sel = self.listbox.curselection()
        if not sel:
            messagebox.showwarning("提示", "请先在检测结果里选择一个印章")
            return
        idx = int(sel[0])

        crops = self.result["crops"]
        if idx < 0 or idx >= len(crops):
            messagebox.showerror("错误", "选择索引无效")
            return

        base = os.path.splitext(os.path.basename(self.img_path))[0]
        default_name = f"{base}_stamp_{idx+1}.png"

        out_path = filedialog.asksaveasfilename(
            title="保存选中的印章",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG", "*.png")]
        )
        if not out_path:
            return

        cv2.imwrite(out_path, crops[idx])
        messagebox.showinfo("完成", f"已保存：\n{out_path}")

    def _show_pil(self, widget: tk.Label, pil_img: Image.Image, kind: str):
        # Fit to a reasonable preview size while keeping aspect ratio
        max_w, max_h = 420, 260
        img = pil_img.copy()
        img.thumbnail((max_w, max_h))
        photo = ImageTk.PhotoImage(img)
        widget.configure(image=photo)

        if kind == "src":
            self._src_photo = photo
        elif kind == "stamp":
            self._stamp_photo = photo
        else:
            self._crop_photo = photo


if __name__ == "__main__":
    App().mainloop()
