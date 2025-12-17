"""
YOLOv11 硬幣檢測 GUI 應用程式
使用 CustomTkinter 建立現代化介面

作者: AI Course
日期: 2024
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import os
import numpy as np

# 設定 CustomTkinter 外觀
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# 硬幣面額對應表
COIN_VALUES = {
    '1h': 1, '1t': 1,
    '5h': 5, '5t': 5,
    '10h': 10, '10t': 10,
    '50h': 50, '50t': 50,
    '0': 0,
    'test': 0,
}


class YOLOApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 視窗設定
        self.title("YOLOv11 硬幣檢測系統")
        self.geometry("1200x800")
        self.minsize(1000, 700)

        # 變數
        self.model = None
        self.model_path = ctk.StringVar(value="尚未載入模型")
        self.conf_threshold = ctk.DoubleVar(value=0.25)
        self.is_running = False
        self.cap = None
        self.current_source = None

        # 建立 UI
        self.create_widgets()

    def create_widgets(self):
        """建立所有 UI 元件"""

        # ===== 左側控制面板 =====
        self.control_frame = ctk.CTkFrame(self, width=300)
        self.control_frame.pack(side="left", fill="y", padx=10, pady=10)
        self.control_frame.pack_propagate(False)

        # 標題
        title_label = ctk.CTkLabel(
            self.control_frame,
            text="YOLOv11 硬幣檢測",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack(pady=(20, 10))

        # ----- 模型設定區 -----
        model_section = ctk.CTkFrame(self.control_frame)
        model_section.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            model_section,
            text="模型設定",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))

        # 選擇模型按鈕
        self.select_model_btn = ctk.CTkButton(
            model_section,
            text="選擇模型檔案 (.pt)",
            command=self.select_model,
            height=40
        )
        self.select_model_btn.pack(fill="x", padx=10, pady=5)

        # 模型路徑顯示
        self.model_label = ctk.CTkLabel(
            model_section,
            textvariable=self.model_path,
            wraplength=250,
            font=ctk.CTkFont(size=11)
        )
        self.model_label.pack(pady=(5, 10))

        # 模型狀態指示燈
        self.model_status = ctk.CTkLabel(
            model_section,
            text="● 模型未載入",
            text_color="red",
            font=ctk.CTkFont(size=12)
        )
        self.model_status.pack(pady=(0, 10))

        # ----- 信心閾值設定 -----
        conf_section = ctk.CTkFrame(self.control_frame)
        conf_section.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            conf_section,
            text="信心閾值",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))

        self.conf_slider = ctk.CTkSlider(
            conf_section,
            from_=0.1,
            to=0.9,
            variable=self.conf_threshold,
            command=self.update_conf_label
        )
        self.conf_slider.pack(fill="x", padx=10, pady=5)

        self.conf_label = ctk.CTkLabel(
            conf_section,
            text=f"閾值: {self.conf_threshold.get():.2f}",
            font=ctk.CTkFont(size=12)
        )
        self.conf_label.pack(pady=(0, 10))

        # ----- 來源選擇區 -----
        source_section = ctk.CTkFrame(self.control_frame)
        source_section.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            source_section,
            text="選擇來源",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))

        # 圖片按鈕
        self.image_btn = ctk.CTkButton(
            source_section,
            text="📷 選擇圖片",
            command=self.select_image,
            height=40,
            state="disabled"
        )
        self.image_btn.pack(fill="x", padx=10, pady=5)

        # 影片按鈕
        self.video_btn = ctk.CTkButton(
            source_section,
            text="🎬 選擇影片",
            command=self.select_video,
            height=40,
            state="disabled"
        )
        self.video_btn.pack(fill="x", padx=10, pady=5)

        # 攝影機按鈕
        self.webcam_btn = ctk.CTkButton(
            source_section,
            text="📹 開啟攝影機",
            command=self.toggle_webcam,
            height=40,
            state="disabled"
        )
        self.webcam_btn.pack(fill="x", padx=10, pady=5)

        # 停止按鈕
        self.stop_btn = ctk.CTkButton(
            source_section,
            text="⏹ 停止",
            command=self.stop_detection,
            height=40,
            fg_color="red",
            hover_color="darkred",
            state="disabled"
        )
        self.stop_btn.pack(fill="x", padx=10, pady=(15, 10))

        # ----- 偵測結果區 -----
        result_section = ctk.CTkFrame(self.control_frame)
        result_section.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            result_section,
            text="偵測結果",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))

        self.result_text = ctk.CTkTextbox(
            result_section,
            height=150,
            font=ctk.CTkFont(size=12)
        )
        self.result_text.pack(fill="x", padx=10, pady=(5, 10))

        # 總金額顯示
        self.total_label = ctk.CTkLabel(
            result_section,
            text="總金額: $0",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="green"
        )
        self.total_label.pack(pady=(0, 10))

        # ===== 右側顯示區 =====
        self.display_frame = ctk.CTkFrame(self)
        self.display_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # 影像顯示標籤
        self.image_label = ctk.CTkLabel(
            self.display_frame,
            text="請載入模型並選擇來源",
            font=ctk.CTkFont(size=16)
        )
        self.image_label.pack(expand=True, fill="both")

    def update_conf_label(self, value):
        """更新信心閾值標籤"""
        self.conf_label.configure(text=f"閾值: {value:.2f}")

    def select_model(self):
        """選擇並載入模型"""
        file_path = filedialog.askopenfilename(
            title="選擇 YOLO 模型檔案",
            filetypes=[("PyTorch 模型", "*.pt"), ("所有檔案", "*.*")]
        )

        if file_path:
            self.load_model(file_path)

    def load_model(self, model_path):
        """載入 YOLO 模型 (在獨立線程中執行)"""
        # 更新 UI 狀態
        self.model_status.configure(text="● 載入中...", text_color="yellow")
        self.select_model_btn.configure(state="disabled")

        # 儲存路徑供線程使用
        self._loading_model_path = model_path

        # 在獨立線程中載入模型
        thread = threading.Thread(target=self._load_model_thread, daemon=True)
        thread.start()

    def _load_model_thread(self):
        """在獨立線程中載入模型"""
        try:
            from ultralytics import YOLO

            model_path = self._loading_model_path
            model = YOLO(model_path)

            # 回到主線程更新 UI
            self.after(0, lambda: self._on_model_loaded(model, model_path))

        except Exception as e:
            # 回到主線程處理錯誤
            self.after(0, lambda: self._on_model_load_error(str(e)))

    def _on_model_loaded(self, model, model_path):
        """模型載入成功的回調 (在主線程執行)"""
        self.model = model
        self.model_path.set(os.path.basename(model_path))
        self.model_status.configure(text="● 模型已載入", text_color="green")
        self.select_model_btn.configure(state="normal")

        # 啟用按鈕
        self.image_btn.configure(state="normal")
        self.video_btn.configure(state="normal")
        self.webcam_btn.configure(state="normal")

        messagebox.showinfo("成功", f"模型載入成功!\n{model_path}")

    def _on_model_load_error(self, error_msg):
        """模型載入失敗的回調 (在主線程執行)"""
        self.model_status.configure(text="● 載入失敗", text_color="red")
        self.select_model_btn.configure(state="normal")
        messagebox.showerror("錯誤", f"模型載入失敗:\n{error_msg}")

    def select_image(self):
        """選擇並處理圖片"""
        file_path = filedialog.askopenfilename(
            title="選擇圖片",
            filetypes=[
                ("圖片檔案", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("所有檔案", "*.*")
            ]
        )

        if file_path:
            self.stop_detection()
            self.process_image(file_path)

    def process_image(self, image_path):
        """處理單張圖片"""
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                messagebox.showerror("錯誤", "無法讀取圖片")
                return

            # 執行偵測
            results = self.model.predict(
                frame,
                conf=self.conf_threshold.get(),
                verbose=False
            )

            # 處理結果
            annotated_frame, coins = self.process_results(results, frame)

            # 顯示結果
            self.display_frame_on_gui(annotated_frame)
            self.update_detection_results(coins)

        except Exception as e:
            messagebox.showerror("錯誤", f"處理圖片時發生錯誤:\n{e}")

    def select_video(self):
        """選擇並處理影片"""
        file_path = filedialog.askopenfilename(
            title="選擇影片",
            filetypes=[
                ("影片檔案", "*.mp4 *.avi *.mov *.mkv"),
                ("所有檔案", "*.*")
            ]
        )

        if file_path:
            self.stop_detection()
            self.start_video(file_path)

    def start_video(self, video_path):
        """開始處理影片"""
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            messagebox.showerror("錯誤", "無法開啟影片")
            return

        self.is_running = True
        self.current_source = "video"
        self.stop_btn.configure(state="normal")

        # 在新執行緒中處理影片
        thread = threading.Thread(target=self.video_loop, daemon=True)
        thread.start()

    def toggle_webcam(self):
        """切換攝影機"""
        if self.is_running and self.current_source == "webcam":
            self.stop_detection()
        else:
            self.stop_detection()
            self.start_webcam()

    def start_webcam(self):
        """開啟攝影機"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("錯誤", "無法開啟攝影機")
            return

        # 設定解析度
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.is_running = True
        self.current_source = "webcam"
        self.webcam_btn.configure(text="📹 關閉攝影機")
        self.stop_btn.configure(state="normal")

        # 在新執行緒中處理攝影機
        thread = threading.Thread(target=self.video_loop, daemon=True)
        thread.start()

    def video_loop(self):
        """影片/攝影機處理迴圈"""
        while self.is_running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                if self.current_source == "video":
                    # 影片結束，重新播放
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            # 攝影機鏡像
            if self.current_source == "webcam":
                frame = cv2.flip(frame, 1)

            # 執行偵測
            results = self.model.predict(
                frame,
                conf=self.conf_threshold.get(),
                verbose=False
            )

            # 處理結果
            annotated_frame, coins = self.process_results(results, frame)

            # 在主執行緒更新 GUI
            self.after(0, lambda f=annotated_frame, c=coins: self.update_gui(f, c))

        self.after(0, self.on_video_stopped)

    def on_video_stopped(self):
        """影片停止時的處理"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.webcam_btn.configure(text="📹 開啟攝影機")
        self.stop_btn.configure(state="disabled")

    def update_gui(self, frame, coins):
        """更新 GUI (在主執行緒)"""
        if self.is_running:
            self.display_frame_on_gui(frame)
            self.update_detection_results(coins)

    def process_results(self, results, frame):
        """處理偵測結果"""
        detected_coins = []
        annotated_frame = frame.copy()

        for result in results:
            boxes = result.boxes
            names = result.names

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = names[cls_id]

                detected_coins.append(class_name)

                # 根據硬幣類型選擇顏色 (BGR 格式，深色以便在白色背景上看清)
                if class_name.startswith('50'):
                    color = (0, 140, 255)       # 深橙色 (50元)
                    text_color = (255, 255, 255)  # 白色文字
                elif class_name.startswith('10'):
                    color = (139, 69, 19)       # 深棕色 (10元)
                    text_color = (255, 255, 255)
                elif class_name.startswith('5'):
                    color = (0, 100, 0)         # 深綠色 (5元)
                    text_color = (255, 255, 255)
                elif class_name.startswith('1'):
                    color = (139, 0, 0)         # 深藍色 (1元)
                    text_color = (255, 255, 255)
                else:
                    color = (128, 0, 128)       # 紫色 (其他)
                    text_color = (255, 255, 255)

                # 繪製邊界框 (加粗)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 4)

                # 繪製標籤
                value = COIN_VALUES.get(class_name, 0)
                label = f"{class_name} {conf:.2f}"
                if value > 0:
                    label += f" (${value})"

                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                # 標籤背景
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - label_h - 10),
                    (x1 + label_w + 10, y1),
                    color, -1
                )
                # 標籤文字
                cv2.putText(
                    annotated_frame, label,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2
                )

        # 顯示總金額 (深紅色，白色背景上清晰可見)
        if detected_coins:
            total = sum(COIN_VALUES.get(c, 0) for c in detected_coins)
            total_text = f"Total: ${total}"
            # 先畫深色背景
            (tw, th), _ = cv2.getTextSize(total_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
            cv2.rectangle(annotated_frame, (5, 10), (tw + 20, th + 25), (0, 0, 139), -1)
            # 再畫白色文字
            cv2.putText(
                annotated_frame, total_text,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3
            )

        return annotated_frame, detected_coins

    def display_frame_on_gui(self, frame):
        """在 GUI 上顯示影像"""
        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 調整大小以適應顯示區域
        display_width = self.display_frame.winfo_width() - 20
        display_height = self.display_frame.winfo_height() - 20

        if display_width > 0 and display_height > 0:
            h, w = frame_rgb.shape[:2]
            scale = min(display_width / w, display_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
        else:
            frame_resized = frame_rgb

        # 轉換為 CTk 可用的格式
        image = Image.fromarray(frame_resized)
        photo = ctk.CTkImage(light_image=image, dark_image=image, size=(frame_resized.shape[1], frame_resized.shape[0]))

        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo

    def update_detection_results(self, coins):
        """更新偵測結果顯示"""
        # 統計各類硬幣數量
        coin_counts = {}
        for coin in coins:
            coin_counts[coin] = coin_counts.get(coin, 0) + 1

        # 更新文字框
        self.result_text.delete("1.0", "end")
        if coin_counts:
            for coin, count in sorted(coin_counts.items()):
                value = COIN_VALUES.get(coin, 0)
                self.result_text.insert("end", f"{coin}: {count} 個")
                if value > 0:
                    self.result_text.insert("end", f" (${value * count})")
                self.result_text.insert("end", "\n")
        else:
            self.result_text.insert("end", "未偵測到硬幣")

        # 計算總金額
        total = sum(COIN_VALUES.get(c, 0) for c in coins)
        self.total_label.configure(text=f"總金額: ${total}")

    def stop_detection(self):
        """停止偵測"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.webcam_btn.configure(text="📹 開啟攝影機")
        self.stop_btn.configure(state="disabled")

    def on_closing(self):
        """關閉視窗時的處理"""
        self.stop_detection()
        self.destroy()


def main():
    """主程式"""
    app = YOLOApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()
