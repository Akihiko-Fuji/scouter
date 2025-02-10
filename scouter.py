#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Clipboard Scouter: A tool for OCR processing of clipboard images
Author: Akihiko Fujita
Version: 1.3

Copyright 2025 Akihiko Fujita

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageGrab, ImageEnhance
import pytesseract
import re
import pyperclip
import configparser
from datetime import datetime
import traceback
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum
import logging
import sys
import locale

# Configure logging / ログの設定
logging.basicConfig(
    filename='ocr_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 最初にエンコードを明示しておく
if hasattr(sys, 'frozen'):
    # Ensure proper encoding for compiled environment
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

class ProcessingMode(Enum):
    """Enumeration for different processing modes with display text
       さまざまな処理モードの列挙とテキスト表示"""
    CALCULATION = ("calculation", "計算式と合計値モード")
    TEXT =        ("text"       , "テキスト抽出モード")
    TABLE =       ("table"      , "表形式モード")

    def __init__(self, value: str, display_text: str):
        self._value_ = value
        self.display_text = display_text

    @classmethod
    def get_display_text(cls, value: str) -> str:
        """Get the display text for a given mode value
           指定されたモードに対応する表示用テキストを取得"""
        for mode in cls:
            if mode.value == value:
                return mode.display_text
        return cls.CALCULATION.display_text  # Default text / 標準テキスト

class PSMMode(Enum):
    """Enumeration for Tesseract PSM modes
       Tesseract PSMモードの列挙"""
    SPARSE_TEXT = "3"   # 文字優先
    UNIFORM_TEXT = "6"  # 標準
    SINGLE_LINE = "11"  # 数値優先

@dataclass
class OCRConfig:
    """Configuration settings for OCR processing
       OCR処理の構成設定"""
    tesseract_cmd: str
    language: str
    enable_logging: bool

    @classmethod
    def from_config_file(cls, config_path: str = "config.ini") -> 'OCRConfig':
        """Create OCRConfig from a configuration file
           設定ファイルからOCRConfigを作成する"""
        config = configparser.ConfigParser()
        default_tesseract = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        
        if os.path.exists(config_path):
            config.read(config_path)
            return cls(
                tesseract_cmd=config.get("OCR", "tesseract_cmd", fallback=default_tesseract),
                language=config.get("OCR", "language", fallback="eng"),
                enable_logging=config.getboolean("OCR", "enable_logging", fallback=True)
            )
        return cls(
            tesseract_cmd=default_tesseract,
            language="eng",
            enable_logging=True
        )

class ImageProcessor:
    """Handles all image processing and OCR operations
       すべての画像処理とOCR操作を行う"""
    def __init__(self, config: OCRConfig):
        self.config = config
        pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Enhance image quality for better OCR results
           より良いOCR結果を得るために画質を向上させる"""
        width, height = image.size
        image = image.resize((width * 2, height * 2), Image.LANCZOS)
        image = image.convert("L")
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(2.0)

    def extract_text(self, image: Image.Image, psm_mode: str) -> str:
        """Extract text from image with improved encoding handling for compiled environment"""
        try:
            processed_image = self.preprocess_image(image)
            custom_config = f'--psm {psm_mode}'
            
            # Force UTF-8 output from Tesseract
            if hasattr(sys, 'frozen'):
                custom_config += ' --encoding UTF8'
            
            raw_text = pytesseract.image_to_string(
                processed_image,
                lang=self.config.language,
                config=custom_config
            )

            # Handle encoding in compiled environment
            if hasattr(sys, 'frozen'):
                try:
                    # First try UTF-8
                    raw_text = raw_text.encode('utf-8', errors='ignore').decode('utf-8')
                except UnicodeError:
                    # Fallback to system locale
                    system_encoding = locale.getpreferredencoding()
                    raw_text = raw_text.encode(system_encoding, errors='ignore').decode('utf-8')

            # Clean up text
            cleaned_text = self._clean_japanese_text(raw_text)
            return cleaned_text

        except Exception as e:
            logging.error(f"OCR extraction error: {str(e)}")
            return ""

    def _clean_japanese_text(self, text: str) -> str:
        """Clean Japanese text with improved character handling"""
        # Remove control characters
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        
        # Keep only valid Japanese characters and basic punctuation
        valid_chars = (
            r'[\u3000-\u303F]'   # Japanese punctuation / 句読点
            r'|[\u3040-\u309F]'  # Hiragana / ひらがな
            r'|[\u30A0-\u30FF]'  # Katakana / カタカナ
            r'|[\u4E00-\u9FFF]'  # Kanji / 漢字
            r'|[\uFF00-\uFFEF]'  # Full-width characters / 全角文字
            r'|[\u0020-\u007E]'  # Basic Latin / 英数
        )
        
        text = ''.join(char for char in text if re.match(valid_chars, char))
        return text

class TextProcessor:
    """Handles text processing operations
       テキスト処理操作を扱う"""
    @staticmethod
    def extract_numbers_and_calculate(text: str) -> Tuple[Optional[str], Optional[int]]:
        """Extract numbers from text and calculate sum
           テキストから数値を抽出し、合計を計算する"""
        # Extract all numbers from the text using a regular expression
        # 正規表現を使用してテキストからすべての数値を抽出
        numbers = re.findall(r'\d+', text)
        if not numbers:
            return None, None  # Return None if no numbers are found / 数値が見つからない場合はNoneを返す
        
        # Convert extracted numbers to integers and calculate their sum / 抽出された数値を整数に変換し、その合計を計算
        numbers = [int(n) for n in numbers]
        return " + ".join(map(str, numbers)), sum(numbers)

    @staticmethod
    def extract_table(text: str) -> str:
        """Convert text to table format with improved encoding handling"""
        try:
            if hasattr(sys, 'frozen'):
                # Handle encoding in compiled environment
                system_encoding = locale.getpreferredencoding()
                try:
                    text = text.encode(system_encoding).decode('utf-8', errors='ignore')
                except UnicodeError:
                    text = text.encode('utf-8', errors='ignore').decode('utf-8')
            
            # Clean and process the text
            lines = text.splitlines()
            processed_lines = []
            
            for line in lines:
                if line.strip():
                    # Clean each line
                    cleaned_line = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', line)
                    valid_chars = (
                        r'[\u3000-\u303F]'   # Japanese punctuation / 句読点
                        r'|[\u3040-\u309F]'  # Hiragana /ひらがな
                        r'|[\u30A0-\u30FF]'  # Katakana / カタカナ
                        r'|[\u4E00-\u9FFF]'  # Kanji / 漢字
                        r'|[\uFF00-\uFFEF]'  # Full-width characters / 全角文字
                        r'|[\u0020-\u007E]'  # Basic Latin / 英数
                    )
                    cleaned_line = ''.join(char for char in cleaned_line if re.match(valid_chars, char))
                    if cleaned_line:
                        processed_lines.append(cleaned_line)
            
            return "\n".join(processed_lines)

        except Exception as e:
            logging.error(f"Table extraction error: {str(e)}")
            return text

class Theme:
    """Manages application theming
       アプリケーションのテーマ設定"""
    DARK = {
        "window_bg": "#2E2E2E",  # アプリ全体のウィンドウ背景色
        "window_fg": "#FFFFFF",  # アプリ全体のウィンドウ前景色（主にラベルのテキスト）
        "text_bg":   "#1E1E1E",  # テキストエリアの背景色（結果表示用）
        "text_fg":   "#FFFFFF",  # テキストエリアの文字色
        "button_bg": "#3E3E3E",  # ボタンの背景色
        "button_fg": "#FFFFFF",  # ボタンの文字色
        "frame_bg":  "#2E2E2E",  # ボタンフレームやその他のフレーム背景色
        "option_bg": "#2E2E2E",  # オプションメニュー（ドロップダウン）の背景色
        "option_fg": "#FFFFFF",  # オプションメニューの文字色
        "menu_bg":   "#2E2E2E",  # ドロップダウンリストの背景色
        "menu_fg":   "#FFFFFF",  # ドロップダウンリストの文字色
        "active_bg": "#4E4E4E",  # ボタンやメニューのホバー時の背景色
        "active_fg": "#FFFFFF"   # ボタンやメニューのホバー時の文字色
    }
    LIGHT = {
        "window_bg": "#F0F0F0",  # 色の指定は上と同じ
        "window_fg": "#000000",
        "text_bg":   "#FFFFFF",
        "text_fg":   "#000000",
        "button_bg": "#E0E0E0",
        "button_fg": "#000000",
        "frame_bg":  "#F0F0F0",
        "option_bg": "#F0F0F0",
        "option_fg": "#000000",
        "menu_bg":   "#F0F0F0",
        "menu_fg":   "#000000",
        "active_bg": "#D0D0D0",
        "active_fg": "#000000"
    }

class OCRWindow:
    """Main application window
       メイン・ウィンドウ"""
    def __init__(self, root: tk.Tk):
        # Initialize configuration and processors / 設定とプロセッサを初期化
        self.config = OCRConfig.from_config_file()
        self.image_processor = ImageProcessor(self.config)
        self.text_processor = TextProcessor()
        self.setup_window(root)
        self.previous_image = None

    def setup_window(self, root: tk.Tk) -> None:
        """Initialize the main window and its components
           メイン・ウィンドウとそのコンポーネントを初期化する"""
        self.root = root
        self.root.title("クリップボード スカウター")  # Set the title of the application / アプリケーションのタイトルを設定
        self.root.geometry("420x420")                 # Set the window size / ウィンドウサイズを設定
        self.root.resizable(False, False)             # Disable resizing / サイズ変更を無効化

        # Variables / 変数
        self.mode = tk.StringVar(value=ProcessingMode.CALCULATION.value)  # Current processing mode / 現在の処理モード
        self.psm_mode = tk.StringVar(value=PSMMode.UNIFORM_TEXT.value)    # Current PSM mode / 現在のPSMモード
        self.dark_mode = tk.BooleanVar(value=False)                       # Dark mode toggle / ダークモードの切り替え

        # Create and arrange widgets / ウィジェットを作成して配置
        self.create_widgets()
        self.setup_layout()
        self.set_theme()
        self.start_clipboard_monitoring()

    def create_widgets(self) -> None:
        """Create all GUI widgets
           すべてのGUIウィジェットを作成する"""
        # Create and set the custom style for the application / アプリケーションのカスタムスタイルを作成して設定する
        self.style = ttk.Style()
        self.style.theme_create("ClipboardScouter", parent="default")
        self.style.theme_use("ClipboardScouter")
        
        # Label for displaying mode description / モードの説明を表示するラベル
        self.label_result = tk.Label(
            self.root,
            text="計算式と合計値モード",
            font=("Arial", 12, "bold")  # Font settings for better readability / 見やすさのためのフォント設定
        )
        
        # ScrolledText widget for displaying results / 結果を表示するためのScrolledTextウィジェット
        self.result_display = ScrolledText(
            self.root,
            font=("Courier New", 16),  # Monospaced font for consistent alignment / 一定の整列を保つモノスペースフォント
            height=10,
            wrap=tk.WORD               # Word wrap for better readability / 読みやすさのための単語単位での折り返し
        )

        # Frame to hold buttons and controls / ボタンやコントロールを保持するフレーム
        self.button_frame = tk.Frame(self.root)
        
        # Create individual widgets for mode controls and copy button / モード切り替えコントロールとコピー用ボタンの個別作成
        self.create_mode_controls()
        self.create_copy_button()

    def create_mode_controls(self) -> None:
        """Create mode selection controls with fixed widths
           固定幅でモード選択コントロールを作成する"""
        # Create a custom style for the option menus / オプションメニュー専用のカスタムスタイルを作成
        self.style.configure(
            "Mode.TMenubutton",
            background="#2E2E2E",  # Background color for the menu button / メニューボタンの背景色
            foreground="#FFFFFF",  # Text color for the menu button / メニューボタンの文字色
            relief="flat",         # Flat appearance / フラットな見た目
            padding=(5, 2)         # Padding for a better click area / クリックエリアを確保するパディング
        )
        
        # Mode toggle button with predefined options / 定義済みオプションを持つモード切り替えボタン
        self.mode_toggle = ttk.OptionMenu(
            self.button_frame,
            self.mode,
            ProcessingMode.CALCULATION.value,          # Default mode / デフォルトのモード
            *[mode.value for mode in ProcessingMode],  # Options from ProcessingMode Enum / ProcessingMode Enumからのオプション
            command=self.update_mode                   # Update mode when selection changes / 選択変更時にモードを更新
        )
        self.mode_toggle.config(width=10)  # Fixed width for consistent UI / 一貫したUIのための固定幅

        # PSM menu for OCR mode selection / OCRモード選択用PSMメニュー
        psm_modes = [
            "3: 文字優先　",  # Prioritize sparse text / 疎な文字を優先
            "6: 標準　　　",  # Standard mode / 標準モード
            "11: 数値優先　"  # Prioritize numbers / 数値を優先
        ]
        self.psm_menu = ttk.OptionMenu(
            self.button_frame,
            self.psm_mode,
            psm_modes[1],     # Default PSM mode / デフォルトのPSMモード
            *psm_modes,
            style="Mode.TMenubutton"    # Use custom style / カスタムスタイルを使用
        )
        self.psm_menu.config(width=12)  # Slightly wider for longer labels / 長いラベルのため少し幅広に設定

        # Configure the dropdown menus for consistent appearance / 一貫した外観のためドロップダウンメニューを設定
        self.configure_dropdown_menu(self.mode_toggle["menu"])
        self.configure_dropdown_menu(self.psm_menu["menu"])

        # Checkbox for dark mode toggle / ダークモード切り替え用チェックボックス
        self.dark_mode_toggle = ttk.Checkbutton(
            self.button_frame,
            text="ダークモード",         # Label text for dark mode / ダークモードのラベルテキスト
            variable=self.dark_mode,
            command=self.toggle_theme,   # Toggle theme on change / 変更時にテーマを切り替え
            style="Toggle.TCheckbutton"  # Custom style for checkboxes / チェックボックス用のカスタムスタイル
        )

    def configure_dropdown_menu(self, menu: tk.Menu) -> None:
        """Configure colors for dropdown menu
           ドロップダウンメニューの色を設定する"""
        theme = Theme.DARK if self.dark_mode.get() else Theme.LIGHT  # Select theme based on current mode / 現在のモードに基づいてテーマを選択
        
        menu.configure(
            bg=theme["option_bg"],                # Background color / 背景色
            fg=theme["option_fg"],                # Text color / 文字色
            activebackground=theme["active_bg"],  # Active item background / アクティブ項目の背景
            activeforeground=theme["active_fg"],  # Active item text color / アクティブ項目の文字色
            relief="flat",                        # Flat style / フラットスタイル
            bd=0                                  # No border / ボーダーなし
        )

    def configure_option_menu_style(self, option_menu: ttk.OptionMenu) -> None:
        """Configure the style for option menus including their dropdowns
           オプションメニューとそのドロップダウンを含むスタイルを設定する"""
        menu = option_menu["menu"]
        theme = Theme.DARK if self.dark_mode.get() else Theme.LIGHT  # Use theme-specific colors / テーマごとの色を使用
        
        # Configure dropdown appearance / ドロップダウンの外観を設定
        menu.configure(
            bg=theme["menu_bg"],                  # Dropdown background / ドロップダウンの背景色
            fg=theme["menu_fg"],                  # Dropdown text color / ドロップダウンの文字色
            activebackground=theme["active_bg"],  # Active item background / アクティブ項目の背景
            activeforeground=theme["active_fg"],  # Active item text color / アクティブ項目の文字色
        )
    def create_copy_button(self) -> None:
        """Create the copy button with consistent styling
           一貫性のあるスタイルでコピー用ボタンを作成する"""
        # Copy button to copy the result to the clipboard / 結果をクリップボードにコピーするためのボタン
        self.copy_button = ttk.Button(      # Changed to ttk.Button for better styling / スタイリング向上のためttk.Buttonを使用
            self.button_frame,
            text="結果コピー",              # Label text for the button / ボタンのラベルテキスト
            style="Copy.TButton",           # Custom style for the copy button / コピー用ボタンのカスタムスタイル
            command=self.copy_to_clipboard  # Callback to copy the result to clipboard / 結果をクリップボードにコピーするコールバック
        )

    def set_theme(self) -> None:
        """Apply the current theme to all widgets
           現在のテーマをすべてのウィジェットに適用する"""
        theme = Theme.DARK if self.dark_mode.get() else Theme.LIGHT  # Select theme based on dark mode state / ダークモードの状態に基づいてテーマを選択
    
        # Configure window and frame backgrounds / ウィンドウとフレームの背景色を設定
        self.root.configure(bg=theme["window_bg"])         # Set main window background / メインウィンドウの背景を設定
        self.button_frame.configure(bg=theme["frame_bg"])  # Set button frame background / ボタンフレームの背景を設定
        self.label_result.configure(bg=theme["window_bg"], fg=theme["window_fg"])  # Update label colors / ラベルの色を更新

        # Configure text display colors / テキスト表示の色を設定
        self.result_display.configure(
            bg=theme["text_bg"],               # Text area background color / テキストエリアの背景色
            fg=theme["text_fg"],               # Text color / テキストの色
            insertbackground=theme["text_fg"]  # Cursor color in text area / テキストエリア内のカーソル色
        )

        # Configure ttk styles for widgets / ウィジェット用のttkスタイルを構成
        self.style.configure(
            "TButton",                       # General button style / 一般的なボタンスタイル
            background=theme["button_bg"],   # Button background / ボタンの背景色
            foreground=theme["button_fg"],   # Button text color / ボタンの文字色
            bordercolor=theme["button_bg"],  # Border color / 境界線の色
            lightcolor=theme["button_bg"],   # Light edge color / 明るいエッジ色
            darkcolor=theme["button_bg"],    # Dark edge color / 暗いエッジ色
            relief="flat"                    # Flat style for buttons / フラットなスタイル
        )

        self.style.configure(
            "Copy.TButton",                  # Specific style for the copy button / コピー用ボタンの特定のスタイル
            background=theme["button_bg"],
            foreground=theme["button_fg"],
            bordercolor=theme["button_bg"],
            relief="flat",                   # Flat style / フラットスタイル
            padding=(10, 5)                  # Padding for larger click area / より大きなクリックエリアのためのパディング
        )

        self.style.configure(
            "TCheckbutton",                  # General checkbutton style / 一般的なチェックボックススタイル
            background=theme["frame_bg"],    # Background color / 背景色
            foreground=theme["window_fg"],   # Text color / 文字色
            relief="flat"                    # Flat style / フラットスタイル
        )

        self.style.configure(
            "Toggle.TCheckbutton",           # Style for the dark mode toggle / ダークモード切り替え用スタイル
            background=theme["frame_bg"],
            foreground=theme["window_fg"],
            relief="flat"
        )

        self.style.configure(
            "Mode.TMenubutton",             # Style for dropdown menus / ドロップダウンメニューのスタイル
            background=theme["option_bg"],  # Dropdown background color / ドロップダウンの背景色
            foreground=theme["option_fg"],  # Dropdown text color / ドロップダウンの文字色
            relief="flat",                  # Flat style / フラットスタイル
            padding=(5, 2),                 # Padding for better click area / より良いクリックエリアのためのパディング
            arrowcolor=theme["option_fg"]   # Arrow icon color / 矢印アイコンの色
        )
        self.mode_toggle.configure(style="Mode.TMenubutton")  # Apply style to mode toggle / モード切り替えにスタイルを適用

        # Define interactions for dropdown menus / ドロップダウンメニューの動作を定義
        self.style.map(
            "Mode.TMenubutton",
            background=[("active", theme["active_bg"]),    # Active background / アクティブ背景
                        ("pressed", theme["active_bg"])],  # Pressed background / 押下時の背景
            foreground=[("active", theme["active_fg"]),    # Active text color / アクティブ文字色
                        ("pressed", theme["active_fg"])]   # Pressed text color / 押下時の文字色
        )

        # Reconfigure dropdown menus based on the current theme / 現在のテーマに基づいてドロップダウンメニューを再設定
        if hasattr(self, 'mode_toggle'):
            self.configure_dropdown_menu(self.mode_toggle["menu"])
        if hasattr(self, 'psm_menu'):
            self.configure_dropdown_menu(self.psm_menu["menu"])
        
        # Update window and frame backgrounds to ensure consistency / ウィンドウとフレームの背景を一貫性を持たせるために更新
        self.root.configure(bg=theme["window_bg"])
        self.button_frame.configure(bg=theme["frame_bg"])
        self.label_result.configure(bg=theme["window_bg"], fg=theme["window_fg"])

        # Configure text display / テキスト表示のスタイルを設定
        self.result_display.configure(
            bg=theme["text_bg"],               # 背景色
            fg=theme["text_fg"],               # 文字色
            insertbackground=theme["text_fg"]  # カーソル色
        )

        # Other existing style configurations / 他のスタイル設定
        self.style.configure(
            "TCheckbutton",
            background=theme["frame_bg"],      # チェックボックスの背景色
            foreground=theme["window_fg"]      # チェックボックスの文字色
        )

        self.style.configure(
            "Copy.TButton",
            background=theme["button_bg"],     # コピー用ボタンの背景色
            foreground=theme["button_fg"]      # コピー用ボタンの文字色
        )

        # Configure the hover/active states / ホバー状態とアクティブ状態のスタイルを設定
        self.style.map("TButton",
            background=[("active", theme["active_bg"])],  # ホバー中の背景色
            foreground=[("active", theme["active_fg"])]   # ホバー中の文字色
        )

        self.style.map("TOptionMenu",
            background=[("active", theme["active_bg"]),   # アクティブな背景色
                       ("pressed", theme["active_bg"])],  # 押下中の背景色
            foreground=[("active", theme["active_fg"]),   # アクティブな文字色
                       ("pressed", theme["active_fg"])]   # 押下中の文字色
        )

        self.style.map("TCheckbutton",
            background=[("active", theme["frame_bg"])],   # チェックボックスの背景色
            foreground=[("active", theme["window_fg"])]   # チェックボックスの文字色
        )

        # Configure the dropdown menu styles / ドロップダウンメニューのスタイルを設定
        self.configure_option_menu_style(self.mode_toggle)  # モード切り替えメニューのスタイル設定
        self.configure_option_menu_style(self.psm_menu)     # PSMメニューのスタイル設定

        # Additional style configurations for fixed-width menus / 固定幅メニューの追加スタイル設定
        self.style.configure(
            "TOptionMenu",
            background=theme["option_bg"],  # メニューの背景色
            foreground=theme["option_fg"],  # メニューの文字色
            relief="flat",                  # フラットな外観
            padding=(5, 2),                 # 余白の設定
            arrowcolor=theme["option_fg"]   # 矢印の色（ダークモードで見やすくするため）
        )

    def update_mode(self, _: str) -> None:
        """Update the display mode and title.
        表示モードとタイトルを更新する。
        
        This method retrieves the current display text based on the selected 
        processing mode, updates the result label, and ensures the dropdown menus 
        retain their style after the mode changes.
        """
        # Get the current mode's display text / 現在のモードの表示テキストを取得する
        current_mode_text = ProcessingMode.get_display_text(self.mode.get())
        logging.debug(f"Mode changed to: {self.mode.get()}, Display text: {current_mode_text}")

        # Update the label text / ラベルの文字を更新
        self.label_result.config(text=current_mode_text)
        self.label_result.update_idletasks()  # Force redraw if necessary / 必要に応じて強制的に再描画

        # Ensure dropdown menus maintain their style after mode change / モード変更後もドロップダウンメニューのスタイルを維持
        if hasattr(self, 'mode_toggle'):
            self.configure_dropdown_menu(self.mode_toggle["menu"])
        if hasattr(self, 'psm_menu'):
            self.configure_dropdown_menu(self.psm_menu["menu"])

    def toggle_theme(self) -> None:
        """Toggle between light and dark themes.
        ライトテーマとダークテーマの切り替えを行う。
        
        This method switches the theme by calling the `set_theme` method.
        """
        self.set_theme()

    def setup_layout(self) -> None:
        """Arrange widgets in the window with consistent padding.
        ウィジェットをウィンドウ内に統一された余白で配置する。
        
        This method organizes all widgets within the main window, ensuring
        consistent spacing and alignment.
        """
        # Configure the result label with padding / 結果ラベルを余白付きで配置
        self.label_result.pack(pady=10)

        # Configure the result display with padding and expand options / 結果表示を余白付きで配置し拡張可能に設定
        self.result_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Configure button frame with padding and full width / ボタンフレームを全幅で余白付きに設定
        self.button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Arrange buttons with consistent spacing / ボタンを一定間隔で配置
        self.mode_toggle.pack(side=tk.LEFT, padx=5)
        self.psm_menu.pack(side=tk.LEFT, padx=5)
        self.dark_mode_toggle.pack(side=tk.LEFT, padx=5)
        self.copy_button.pack(side=tk.RIGHT, padx=5)

    def process_clipboard_image(self, image: Image.Image) -> None:
        """Process clipboard image according to the current mode.
        現在のモードに応じてクリップボード画像を処理する。
        
        This method extracts text from the image and processes it based on the selected mode:
        - Calculation: Calculates totals from numbers in the text.
        - Table: Extracts tabular data.
        - Text: Displays the raw text.
        """
        # Extract text from the clipboard image / クリップボード画像からテキストを抽出
        text = self.image_processor.extract_text(image, self.psm_mode.get().split(":")[0])

        # Check the encoding of OCR results / OCR結果のエンコードを確認
        try:
            text.encode("utf-8").decode("utf-8")  # 正常なら何も起きない
        except UnicodeDecodeError:
            logging.warning("OCR結果が文字化けしている可能性があります。適切に修正します。")
            text = text.encode("shift_jis").decode("utf-8", errors="replace")

        if self.mode.get() == ProcessingMode.CALCULATION.value:
            # Handle calculation mode / 計算モードの処理
            formula, total = self.text_processor.extract_numbers_and_calculate(text)
            if formula and total is not None:
                content = f"{formula} = {total}"
                self.update_display(content)
                self.log_result("calculation", content)
            else:
                self.update_display("数字が見つかりません")

        elif self.mode.get() == ProcessingMode.TABLE.value:
            # Handle table mode / テーブルモードの処理
            table_content = self.text_processor.extract_table(text)
            self.update_display(table_content)
            self.log_result("table", table_content)

        else:  # Text mode / テキストモード
            self.update_display(text.strip())
            self.log_result("text", text.strip())
            logging.debug(f"OCR extracted text: {text}")

    def update_display(self, content: str) -> None:
        """Update the display with new content.
        新しいコンテンツでディスプレイを更新する。
        
        Clears the result display and inserts the provided content.
        """
        self.result_display.delete(1.0, tk.END)  # Clear the display / 表示をクリア
        self.result_display.insert(tk.END, content)  # Insert new content / 新しい内容を挿入

    def copy_to_clipboard(self) -> None:
        """Copy the processed result to the clipboard.
        処理結果をクリップボードにコピーする。
        
        Depending on the mode, this method copies either the calculation result or
        the plain content to the clipboard.
        """
        # Get content from the result display / 結果表示からコンテンツを取得
        content = self.result_display.get("1.0", tk.END).strip()
        
        if self.mode.get() == ProcessingMode.CALCULATION.value:
            # Extract and copy the calculated result / 計算結果を抽出してコピー
            match = re.search(r"=\s*(\d+)", content)
            pyperclip.copy(match.group(1) if match else "")
        else:
            # Copy the entire content / 全体をコピー
            pyperclip.copy(content)

    def start_clipboard_monitoring(self) -> None:
        """Start monitoring the clipboard for images.
        クリップボードの画像監視を開始する。
        
        This method initiates the clipboard monitoring process, which checks for
        new images in the clipboard and processes them if detected.
        """
        self.update_clipboard()

    def update_clipboard(self) -> None:
        """Check clipboard for new images and process them.
        クリップボードに新しい画像がないかチェックし、処理する。

        This method continuously monitors the clipboard for new images. 
        If a new image is detected, it processes the image using the appropriate 
        mode (text, table, or calculation).
        """
        try:
            # Grab the current clipboard content / 現在のクリップボード内容を取得
            image = ImageGrab.grabclipboard()
            # Check if the clipboard content is an image and not previously processed / クリップボード内容が画像で未処理の場合
            if isinstance(image, Image.Image) and image != self.previous_image:
                self.previous_image = image          # Update the last processed image / 最後に処理した画像を更新
                self.process_clipboard_image(image)  # Process the new image / 新しい画像を処理
        except Exception as e:
            # Log errors during clipboard monitoring / クリップボード監視中のエラーをログに記録
            logging.error(f"Clipboard monitoring error: {str(e)}")
        
        # Schedule the next clipboard check after 500ms / 500ms後に次のクリップボードチェックをスケジュール
        self.root.after(500, self.update_clipboard)

    def log_result(self, mode: str, content: str) -> None:
        """Log processing results if logging is enabled.
        ロギングが有効な場合、処理結果をログに記録する。

        Parameters:
        mode (str): The processing mode (e.g., text, table, calculation).
        content (str): The content to log.
        """
        # Check if logging is enabled in the configuration / 設定でロギングが有効かを確認
        if self.config.enable_logging:
            try:
                # UTF-8にエンコードされているか確認し、不正な場合は修正
                content = content.encode("utf-8").decode("utf-8")
            except UnicodeDecodeError:
                logging.warning("文字化けが発生しているため修正中")
                content = content.encode("shift_jis").decode("utf-8", errors="replace")

            # ログファイルに追記
            with open("ocr_log.txt", "a", encoding="utf-8") as log_file:
                log_file.write(f"Mode: {mode}\n{content}\n")
            logging.info(f"Mode: {mode}\n{content}")

def main():
    """Main application entry point.
    アプリケーションのエントリーポイント。

    This function initializes the Tkinter root window and starts the main event loop. 
    Any exceptions during the application's execution are logged.
    """
    try:
        # Initialize the Tkinter root window / Tkinterのルートウィンドウを初期化
        root = tk.Tk()
        # Create an instance of the OCRWindow class / OCRWindowクラスのインスタンスを作成
        app = OCRWindow(root)
        # Start the main event loop / メインイベントループを開始
        root.mainloop()
    except Exception as e:
        # Log application errors with a detailed traceback / 詳細なトレースバック付きでアプリケーションエラーをログに記録
        logging.error(f"Application error:\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()
