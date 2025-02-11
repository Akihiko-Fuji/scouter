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
import win32clipboard

# Configure logging / ログの設定
logging.basicConfig(
    filename='ocr_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Ensure proper encoding / 適切なエンコーディングを明示しておく
if hasattr(sys, 'frozen'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

class ProcessingMode(Enum):
    """Enumeration for different processing modes with display text /さまざまな処理モードの列挙とテキスト表示"""
    CALCULATION = ("calculation", "計算式と合計値モード")
    TEXT =        ("text"       , "テキスト抽出モード")
    TABLE =       ("table"      , "表形式モード")

    def __init__(self, value: str, display_text: str):
        self._value_ = value
        self.display_text = display_text

    @classmethod
    def get_display_text(cls, value: str) -> str:
        """Get the display text for a given mode value / 指定されたモードに対応する表示用テキストを取得"""

        for mode in cls:
            if mode.value == value:
                return mode.display_text
        return cls.CALCULATION.display_text

class PSMMode(Enum):
    """ Enumeration for Tesseract PSM modes / Tesseract PSMモードの列挙 """

    SPARSE_TEXT =  "3"  # 文字優先
    UNIFORM_TEXT = "6"  # 標準
    SINGLE_LINE = "11"  # 数値優先

@dataclass
class OCRConfig:
    """Configuration settings for OCR processing / OCR処理の構成設定"""

    tesseract_cmd: str
    language: str
    enable_logging: bool

    @classmethod
    def from_config_file(cls, config_path: str = "config.ini") -> 'OCRConfig':
        """Create OCRConfig from a configuration file /  設定ファイルからOCRConfigを作成する"""

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
    """Handles all image processing and OCR operations / すべての画像処理とOCR操作を行う"""

    def __init__(self, config: OCRConfig):
        self.config = config
        pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Enhance image quality for better OCR results / OCR結果を得るために画質を向上させる"""

        width, height = image.size
        image = image.resize((width * 2, height * 2), Image.LANCZOS)
        image = image.convert("L")
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(2.0)

    def extract_text(self, image: Image.Image, psm_mode: str) -> str:
        """Extract text from image with improved encoding handling for compiled environment / 画像からのテキスト抽出"""

        try:
            processed_image = self.preprocess_image(image)
            custom_config = f'--psm {psm_mode}'
            
            # Force UTF-8 output from Tesseract / TesseractからのUTF-8出力を強制
            if hasattr(sys, 'frozen'):
                custom_config += ' --encoding UTF8'
            
            raw_text = pytesseract.image_to_string(
                processed_image,
                lang=self.config.language,
                config=custom_config
            )

            # Handle encoding in compiled environment / コンパイルされた環境でのエンコード処理
            if hasattr(sys, 'frozen'):
                try:
                    # First try UTF-8
                    raw_text = raw_text.encode('utf-8', errors='ignore').decode('utf-8')
                except UnicodeError:
                    # Fallback to system locale
                    system_encoding = locale.getpreferredencoding()
                    raw_text = raw_text.encode(system_encoding, errors='ignore').decode('utf-8')

            # Clean up text / テキストをクリーンアップ
            cleaned_text = self._clean_japanese_text(raw_text)
            return cleaned_text

        except Exception as e:
            logging.error(f"OCR extraction error: {str(e)}")
            return ""

    def _clean_japanese_text(self, text: str) -> str:
        """Clean Japanese text with improved character handling / 日本語テキストをクリーンアップ"""

        # Remove control characters / 制御文字を削除
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        
        # Keep only valid Japanese characters and basic punctuation / 有効な日本語文字と基本的な句読点のみを保持
        valid_chars = (
            r'[\u3000-\u303F]'   # Japanese punctuation  / 句読点
            r'|[\u3040-\u309F]'  # Hiragana              / ひらがな
            r'|[\u30A0-\u30FF]'  # Katakana              / カタカナ
            r'|[\u4E00-\u9FFF]'  # Kanji                 / 漢字
            r'|[\uFF00-\uFFEF]'  # Full-width characters / 全角文字
            r'|[\u0020-\u007E]'  # Basic Latin           / 英数
        )
        
        text = ''.join(char for char in text if re.match(valid_chars, char))
        return text

class TextProcessor:
    """Handles text processing operations / テキスト処理操作を扱う"""

    @staticmethod
    def extract_numbers_and_calculate(text: str) -> Tuple[Optional[str], Optional[int]]:
        """Extract numbers from text and calculate sum / テキストから数値を抽出し、合計を計算する"""

        # Extract all numbers from the text using a regular expression  / 正規表現を使用してテキストからすべての数値を抽出
        numbers = re.findall(r'\d+', text)
        if not numbers:
            return None, None  # Return None if no numbers are found    / 数値が見つからない場合はNoneを返す
        
        # Convert extracted numbers to integers and calculate their sum / 抽出された数値を整数に変換し、その合計を計算
        numbers = [int(n) for n in numbers]
        return " + ".join(map(str, numbers)), sum(numbers)

    @staticmethod
    def extract_table(text: str) -> str:
        """Convert text to table format with improved encoding handling / テキストをテーブル形式に変換し、エンコード処理を向上させる"""

        try:
            if hasattr(sys, 'frozen'):
                # Handle encoding in compiled environment / コンパイル環境でのエンコード処理
                system_encoding = locale.getpreferredencoding()
                try:
                    text = text.encode(system_encoding).decode('utf-8', errors='ignore')
                except UnicodeError:
                    text = text.encode('utf-8', errors='ignore').decode('utf-8')
            
            # Clean and process the text / テキストを整理し処理
            lines = text.splitlines()
            processed_lines = []
            
            for line in lines:
                if line.strip():
                    # Clean each line / 各行を整理
                    cleaned_line = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', line)
                    valid_chars = (
                        r'[\u3000-\u303F]'   # Japanese punctuation  / 句読点
                        r'|[\u3040-\u309F]'  # Hiragana              / ひらがな
                        r'|[\u30A0-\u30FF]'  # Katakana              / カタカナ
                        r'|[\u4E00-\u9FFF]'  # Kanji                 / 漢字
                        r'|[\uFF00-\uFFEF]'  # Full-width characters / 全角文字
                        r'|[\u0020-\u007E]'  # Basic Latin           / 英数
                    )
                    cleaned_line = ''.join(char for char in cleaned_line if re.match(valid_chars, char))
                    if cleaned_line:
                        processed_lines.append(cleaned_line)
            
            return "\n".join(processed_lines)

        except Exception as e:
            logging.error(f"Table extraction error: {str(e)}")
            return text

class Theme:
    """Manages application theming / アプリケーションのテーマ設定"""

    DARK = {
        "window_bg": "#2E2E2E",  # Background color of the app window                 / アプリのウィンドウ背景色
        "window_fg": "#FFFFFF",  # Foreground color of the app window                 / アプリのウィンドウ前景色（主にラベルのテキスト）
        "text_bg":   "#1E1E1E",  # Background color of text area                      / テキストエリアの背景色（結果表示用）
        "text_fg":   "#FFFFFF",  # Text color of text area                            / テキストエリアの文字色
        "button_bg": "#3E3E3E",  # Background color of buttons                        / ボタンの背景色
        "button_fg": "#FFFFFF",  # Text color of buttons                              / ボタンの文字色
        "frame_bg":  "#2E2E2E",  # Background color of button frames and other frames / ボタンフレームやその他のフレーム背景色
        "option_bg": "#2E2E2E",  # Background color of option menus                   / オプションメニュー（ドロップダウン）の背景色
        "option_fg": "#FFFFFF",  # Text color of option menus                         / オプションメニューの文字色
        "menu_bg":   "#2E2E2E",  # Background color of dropdown lists                 / ドロップダウンリストの背景色
        "menu_fg":   "#FFFFFF",  # Text color of dropdown lists                       / ドロップダウンリストの文字色
        "active_bg": "#4E4E4E",  # Background color when hovering buttons or menus    / ボタンやメニューのホバー時の背景色
        "active_fg": "#FFFFFF"   # Text color when hovering buttons or menus          / ボタンやメニューのホバー時の文字色
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
    """Main application window / メインウィンドウ"""
    
    def __init__(self, root: tk.Tk):
        # Initialize configuration and processors / 設定とプロセッサを初期化
        self.config = OCRConfig.from_config_file()
        self.image_processor = ImageProcessor(self.config)
        self.text_processor = TextProcessor()
        self.setup_window(root)
        self.previous_image = None

    def setup_window(self, root: tk.Tk) -> None:
        """Initialize the main window and its components / メイン・ウィンドウを初期化する"""

        self.root = root
        self.root.title("クリップボード スカウター")  # Set the title of the application / アプリケーションのタイトルを設定
        self.root.geometry("420x420")                 # Set the window size              / ウィンドウサイズを設定
        self.root.resizable(False, False)             # Disable resizing                 / サイズ変更を無効化

        # Variables / 変数
        self.mode = tk.StringVar(value=ProcessingMode.CALCULATION.value)  # Current processing mode / 現在の処理モード
        self.psm_mode = tk.StringVar(value=PSMMode.UNIFORM_TEXT.value)    # Current PSM mode        / 現在のPSMモード
        self.dark_mode = tk.BooleanVar(value=False)                       # Dark mode toggle        / ダークモードの切り替え

        # Create and arrange widgets / ウィジェットを作成して配置
        self.create_widgets()
        self.setup_layout()
        self.set_theme()
        self.start_clipboard_monitoring()

    def create_widgets(self) -> None:
        """Create all GUI widgets / すべてのGUIウィジェットを作成する"""

        # Create and set the custom style for the application                     / アプリケーションのカスタムスタイルを作成して設定する
        self.style = ttk.Style()
        self.style.theme_create("ClipboardScouter", parent="default")
        self.style.theme_use("ClipboardScouter")
        
        # Label for displaying mode description                                   / モードの説明を表示するラベル
        self.label_result = tk.Label(
            self.root,
            text="計算式と合計値モード",
            font=("Arial", 12, "bold")  # Font settings for better readability    / 見やすさのためのフォント設定
        )
        
        # ScrolledText widget for displaying results                              / 結果を表示するためのScrolledTextウィジェット
        self.result_display = ScrolledText(
            self.root,
            font=("Courier New", 16),  # Monospaced font for consistent alignment / 一定の整列を保つモノスペースフォント
            height=10,
            wrap=tk.WORD               # Word wrap for better readability         / 読みやすさのための単語単位での折り返し
        )

        # Frame to hold buttons and controls                                      / ボタンやコントロールを保持するフレーム
        self.button_frame = tk.Frame(self.root)
        
        # Create individual widgets for mode controls and copy button             / モード切り替えコントロールとコピー用ボタンの個別作成
        self.create_mode_controls()
        self.create_copy_button()

    def create_mode_controls(self) -> None:

        # Create a custom style for the option menus / オプションメニュー専用のカスタムスタイルを作成
        self.style.configure(
            "Mode.TMenubutton",
            background="#2E2E2E",  # Background color for the menu button / メニューボタンの背景色
            foreground="#FFFFFF",  # Text color for the menu button       / メニューボタンの文字色
            relief="flat",         # Flat appearance                      / フラットな見た目
            padding=(5, 2)         # Padding for a better click area      / クリックエリアを確保するパディング
        )
        
        # Mode toggle button with predefined options / 定義済みオプションを持つモード切り替えボタン
        self.mode_toggle = ttk.OptionMenu(
            self.button_frame,
            self.mode,
            ProcessingMode.CALCULATION.value,          # Set default value directly
            *[mode.value for mode in ProcessingMode],  # Options from ProcessingMode Enum   / ProcessingMode Enumからのオプション
            command=self.update_mode                   # Update mode when selection changes / 選択変更時にモードを更新
        )
        self.mode_toggle.config(width=10)  # Fixed width for consistent UI / 一貫したUIのための固定幅

        # PSM menu for OCR mode selection / OCRモード選択用PSMメニュー
        psm_modes = [
            "3: 文字優先　",  # Prioritize sparse text / 疎な文字を優先
            "6: 標準　　　",  # Standard mode          / 標準モード
            "11: 数値優先　"  # Prioritize numbers     / 数値を優先
        ]
        self.psm_menu = ttk.OptionMenu(
            self.button_frame,
            self.psm_mode,
            psm_modes[1],               # Default PSM mode / デフォルトのPSMモード
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
            text="ダークモード",         # Label text for dark mode    / ダークモードのラベルテキスト
            variable=self.dark_mode,
            command=self.toggle_theme,   # Toggle theme on change      / 変更時にテーマを切り替え
            style="Toggle.TCheckbutton"  # Custom style for checkboxes / チェックボックス用のカスタムスタイル
        )

    def configure_dropdown_menu(self, menu: tk.Menu) -> None:
        """Configure colors for dropdown menu / ドロップダウンメニューの色を設定する"""

        theme = Theme.DARK if self.dark_mode.get() else Theme.LIGHT  # Select theme based on current mode / 現在のモードに基づいてテーマを選択
        
        menu.configure(
            bg=theme["option_bg"],                # Background color       / 背景色
            fg=theme["option_fg"],                # Text color             / 文字色
            activebackground=theme["active_bg"],  # Active item background / アクティブ項目の背景
            activeforeground=theme["active_fg"],  # Active item text color / アクティブ項目の文字色
            relief="flat",                        # Flat style             / フラットなスタイル
            bd=0                                  # No border              / ボーダーなし
        )

    def configure_option_menu_style(self, option_menu: ttk.OptionMenu) -> None:
        """Configure the style for option menus including their dropdowns / オプションメニューとそのドロップダウンを含むスタイルを設定する"""

        menu = option_menu["menu"]
        theme = Theme.DARK if self.dark_mode.get() else Theme.LIGHT  # Use theme-specific colors / テーマごとの色を使用
        
        # Configure dropdown appearance / ドロップダウンの外観を設定
        menu.configure(
            bg=theme["menu_bg"],                  # Dropdown background    / ドロップダウンの背景色
            fg=theme["menu_fg"],                  # Dropdown text color    / ドロップダウンの文字色
            activebackground=theme["active_bg"],  # Active item background / アクティブ項目の背景
            activeforeground=theme["active_fg"],  # Active item text color / アクティブ項目の文字色
        )
    def create_copy_button(self) -> None:
        """Create the copy button with consistent styling / 一貫性のあるスタイルでコピー用ボタンを作成する"""

        # Copy button to copy the result to the clipboard                              / 結果をクリップボードにコピーするためのボタン
        self.copy_button = ttk.Button(      # Changed to ttk.Button for better styling / スタイリング向上のためttk.Buttonを使用
            self.button_frame,
            text="結果コピー",              # Label text for the button                / ボタンのラベルテキスト
            style="Copy.TButton",           # Custom style for the copy button         / コピー用ボタンのカスタムスタイル
            command=self.copy_to_clipboard  # Callback to copy the result to clipboard / 結果をクリップボードにコピーするコールバック
        )

    def set_theme(self) -> None:
        """Apply the current theme to all widgets / 現在のテーマをすべてのウィジェットに適用する"""

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
            "TButton",                       # General button style   / 一般的なボタンスタイル
            background=theme["button_bg"],   # Button background      / ボタンの背景色
            foreground=theme["button_fg"],   # Button text color      / ボタンの文字色
            bordercolor=theme["button_bg"],  # Border color           / 境界線の色
            lightcolor=theme["button_bg"],   # Light edge color       / 明るいエッジ色
            darkcolor=theme["button_bg"],    # Dark edge color        / 暗いエッジ色
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
            background=theme["frame_bg"],    # Background color          / 背景色
            foreground=theme["window_fg"],   # Text color                / 文字色
            relief="flat"                    # Flat style                / フラットスタイル
        )

        self.style.configure(
            "Toggle.TCheckbutton",           # Style for the dark mode toggle / ダークモード切り替え用スタイル
            background=theme["frame_bg"],
            foreground=theme["window_fg"],
            relief="flat"
        )

        self.style.configure(
            "Mode.TMenubutton",             # Style for dropdown menus      / ドロップダウンメニューのスタイル
            background=theme["option_bg"],  # Dropdown background color     / ドロップダウンの背景色
            foreground=theme["option_fg"],  # Dropdown text color           / ドロップダウンの文字色
            relief="flat",                  # Flat style                    / フラットスタイル
            padding=(5, 2),                 # Padding for better click area / より良いクリックエリアのためのパディング
            arrowcolor=theme["option_fg"]   # Arrow icon color              / 矢印アイコンの色
        )
        self.mode_toggle.configure(style="Mode.TMenubutton")  # Apply style to mode toggle / モード切り替えにスタイルを適用

        # Define interactions for dropdown menus / ドロップダウンメニューの動作を定義
        self.style.map(
            "Mode.TMenubutton",
            background=[("active", theme["active_bg"]),    # Active background  / アクティブ背景
                        ("pressed", theme["active_bg"])],  # Pressed background / 押下時の背景
            foreground=[("active", theme["active_fg"]),    # Active text color  / アクティブ文字色
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
        """Update the display mode and title. / 表示モードとタイトルを更新する"""

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
        """Toggle between light and dark themes. / ライトテーマとダークテーマの切り替えを行う"""
        self.set_theme()

    def setup_layout(self) -> None:
        """Arrange widgets in the window with consistent padding. / ウィジェットをウィンドウ内に統一された余白で配置する"""
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
        """Process clipboard image according to the current mode. / 現在のモードに応じてクリップボード画像を処理する"""
        try:
            # Verify image is valid before processing                  / 処理前に画像が有効か確認
            if not image or not hasattr(image, 'size'):
                return
            
            # Check minimum image size to avoid processing tiny images / 最小画像サイズを確認し、小さすぎる画像の処理を回避
            if image.size[0] < 10 or image.size[1] < 10:
                return
            
            # Proceed with existing OCR processing                     / 既存のOCR処理を続行
            text = self.image_processor.extract_text(image, self.psm_mode.get().split(":")[0])
        
            # Process according to mode                                / モードに応じて処理
            if self.mode.get() == ProcessingMode.CALCULATION.value:
                formula, total = self.text_processor.extract_numbers_and_calculate(text)
                if formula and total is not None:
                    content = f"{formula} = {total}"
                    self.update_display(content)
                    self.log_result("calculation", content)
                else:
                    self.update_display("数字が見つかりません")
                
            elif self.mode.get() == ProcessingMode.TABLE.value:
                table_content = self.text_processor.extract_table(text)
                self.update_display(table_content)
                self.log_result("table", table_content)
            
            else:  # Text mode
                self.update_display(text.strip())
                self.log_result("text", text.strip())
            
        except Exception as e:
            logging.error(f"Image processing error: {str(e)}")
            self.update_display("画像処理中にエラーが発生しました")

    def update_display(self, content: str) -> None:
        """Update the display with new content. / 新しいコンテンツでディスプレイを更新する"""
        self.result_display.delete(1.0, tk.END)  # Clear the display      / 表示をクリア
        self.result_display.insert(tk.END, content)  # Insert new content / 新しい内容を挿入

    def copy_to_clipboard(self) -> None:
        """Copy the processed result to the clipboard. / 処理結果をクリップボードにコピー"""
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
        """
        This method initiates the clipboard monitoring process, which checks for
        new images in the clipboard and processes them if detected.
        クリップボード監視プロセスを開始する。画像検出された場合はOCR処理する
        """

        self.update_clipboard()

    def update_clipboard(self) -> None:
        """
        Check clipboard for new images and process them. / クリップボードに新しい画像がないかチェックする

        1. Quick format check before processing          / 1. 処理前のクイック・フォーマット・チェック
        2. Skipping non-image content                    / 2. 非画像コンテンツのスキップ
        3. Preventing redundant processing               / 3. 冗長処理の防止
        3. Preventing redundant processing
        """
        try:
            # First check if clipboard contains image data without actually grabbing it
            # 最初に実際に画像を取得することなく、クリップボードに画像データが含まれているか確認
            if not self.has_image_in_clipboard():
                self.root.after(500, self.update_clipboard)
                return

            # Only grab the image if we confirmed it exists / 画像が確認された場合にのみ画像を取得
            image = ImageGrab.grabclipboard()
        
            # Verify it's a valid PIL Image and not previously processed
            # 有効なPILイメージであり、以前に処理されていないことを確認
            if isinstance(image, Image.Image) and image != self.previous_image:
                self.previous_image = image
                self.process_clipboard_image(image)

            # Log errors during clipboard monitoring / クリップボード監視中のエラーをログに記録
        except Exception as e:
            logging.error(f"Clipboard monitoring error: {str(e)}")

        # Schedule the next clipboard check after 1000ms / 1000ms後に次のクリップボードチェックをスケジュール
        self.root.after(1000, self.update_clipboard)

    def has_image_in_clipboard(self) -> bool:
        """True if clipboard contains image data, False otherwise / クリップボードに画像データが含まれている場合はTrue、そうでない場合はFalse"""
        try:
            import win32clipboard
        
            # Open clipboard for reading         / 読み取りのためにクリップボードを開く
            win32clipboard.OpenClipboard(None)
        
            try:
                # Check for common image formats / 一般的な画像形式を確認
                formats = []
                format_id = 0
            
                # Get all available formats in clipboard / クリップボード内のすべての利用可能な形式を取得
                while True:
                    format_id = win32clipboard.EnumClipboardFormats(format_id)
                    if not format_id:
                        break
                    formats.append(format_id)
            
                # Define image format identifiers / 画像形式識別子を定義
                image_formats = {
                    win32clipboard.CF_DIB,        # Device Independent Bitmap          / デバイス非依存ビットマップ
                    win32clipboard.CF_BITMAP,     # Windows Bitmap                     / Windowsビットマップ
                    8,                            # CF_DIB (Device Independent Bitmap) / デバイス独立ビットマップ
                    17,                           # CF_DIBV5 (Extended DIB)            / 拡張DIB
                    0x0142,                       # Art[ip]BITMAP
                    0x0319                        # PNG
                }
            
                # Check if any image format is present / 画像形式が存在するか確認
                return bool(set(formats) & image_formats)
            
            finally:
                win32clipboard.CloseClipboard()
            
        except Exception as e:
            logging.debug(f"Clipboard format check error: {str(e)}")
            return False  # On error, return False to be safe / エラーが発生した場合、安全のためFalseを返す


    def log_result(self, mode: str, content: str) -> None:
        """Log processing results if logging is enabled. / ロギングが有効な場合、処理結果をログに記録する"""
        # Check if logging is enabled in the configuration / 設定でロギングが有効かを確認
        if self.config.enable_logging:
            try:
                # Ensure content is encoded in UTF-8, fix if corrupted / コンテンツがUTF-8でエンコードされているか確認し、不正な場合は修正
                content = content.encode("utf-8").decode("utf-8")
            except UnicodeDecodeError:
                logging.warning("文字化けが発生しているため修正中")
                content = content.encode("shift_jis").decode("utf-8", errors="replace")

            # Append log to a file / ログファイルに追記
            with open("ocr_log.txt", "a", encoding="utf-8") as log_file:
                log_file.write(f"Mode: {mode}\n{content}\n")
            logging.info(f"Mode: {mode}\n{content}")

def main():
    """
    This function initializes the Tkinter root window and starts the main event loop.
    Any exceptions during the application's execution are logged.
    Tkinter ルートウィンドウを初期化し、メインイベントループを開始。
    """

    try:
        # Initialize the Tkinter root window        / Tkinterのルートウィンドウを初期化
        root = tk.Tk()
        # Create an instance of the OCRWindow class / OCRWindowクラスのインスタンスを作成
        app = OCRWindow(root)
        # Start the main event loop                 / メインイベントループを開始
        root.mainloop()

    except Exception as e:
        # Log application errors with a detailed traceback / 詳細なトレースバック付きでアプリケーションエラーをログに記録
        logging.error(f"Application error:\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()
