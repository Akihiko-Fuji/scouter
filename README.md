# Abstract:

Clipboard Scouter is an OCR (Optical Character Recognition) processing tool designed to extract and analyze text and numbers from images copied to the clipboard.
It offers various processing modes including text extraction, calculation-based operations, and table conversion.
Users can interact with the tool through a GUI developed using Python's Tkinter library, where they can select the processing mode, view results.
The application utilizes the Tesseract OCR engine for text recognition, with customization options for OCR modes and language settings.
It also features logging capabilities to track errors and operations, and supports configuration through an INI file.
The tool is particularly useful for processing text and numbers from screenshots or clipboard images, performing automatic calculations, and formatting output into tables.

Text embedded in images.
Text you want to search for to solve a problem but can't copy such as a VBA error window,and tables secretly screen-captured in a web conference.
And we can convert material uploaded to speakerdeck.com into text.

# Prerequisites
This program uses an OCR library called Tesseract, so please download it from the URL below and install it beforehand.
for Windows see https://github.com/UB-Mannheim/tesseract/wiki

During installation, several checks must be made to handle the Japanese language.

Select additional script data
 When the following screen appears, click the + for Additional script data (download) to expand it.
  
Select additional language data:
 Click “+” under “Additional language data (download)” to expand it.
 Check the following Japanese ~ item from the expanded items.

 The default installation location (C:\Program Files\Tesseract-OCR\tesseract.exe) is recommended, but the reference location can be changed in the configuration file.

# Files required for the program to work<BR>
Scouter.exe <BR>
config.ini<BR>

No folder is specified. There is no installer. No registry entries. Place the above files in a single folder.

# Description of config.ini
The following three settings can be made in the configuration file. Normally, no changes are required.<BR>
tesseract_cmd = C:\Program Files\Tesseract-OCR\tesseract.exe<BR>
Specify the location where Tesseract is saved for operation. If you did not change the location during installation, you do not need to change this value.<BR>

language = jpn<BR>
Specify the language used for OCR processing. The standard language is Japanese (jpn).
For English documents, set this to eng. For documents that contain both English and Japanese, you can specify jpn+eng, etc.

enable_logging = False<BR>
This mode records the results of OCR processing in a log file. If you want to keep a history of data processing, including checking for errors, change this to True.

# Usage and Functions
![demo](https://github.com/Akihiko-Fuji/scouter/blob/main/demo.gif?raw=true)

Demonstration combined with snipping tool, a standard Windows screen capture.
 Double-click the executable file with the icon to start it.
 
1. Reading result<BR>
　Displays the result of OCR processing based on the image in the clipboard.

  calculation formula and total value mode ... which numbers were extracted, value & total value of calculation<BR>
  text extraction mode ... expands the extracted text string<BR>
  tabular mode ... which numbers are extracted, values & comma delimited<BR>

2. Mode setting<BR>
  You can select the data processing mode. See 1. for mode and processing.

3. OCR Processing<BR>
  You can select the OCR reading processing method. This utilizes Tesseract's PSM function for switching the reading method. If the values cannot be read properly, try changing this processing.

4. Dark mode switching<BR>
  You can change to the dark mode, which is less noticeable even if the Window is left open.

5. Copying the result<BR>
  This function copies the read result to the clipboard. Text extraction mode and tabular mode 1. copy the reading results as they are.
  In the calculation formula and total value mode, the calculation formula is not copied, but “only the total value of the calculation result” is copied.

# Download URL:<BR>
https://github.com/Akihiko-Fuji/scouter/raw/refs/heads/main/scouter.zip

# Important point<BR>
  The program is designed for Japanese. The comments of the program are written in both English and Japanese, so the program can be used in other languages if you modify the UI, such as menus.

# Version<BR>
ver 1.3<BR>
Since Japanese results were sometimes garbled, the decoding process judgment of character codes was reviewed.<BR>
In addition, to solve the problem of program processing stopping when a large amount of data other than images is copied to the clipboard, such as when a large number of cells are copied in Excel, a process to detect the type of data pasted to the clipboard before processing was added.<BR>
Added more comments during the python program. I usually do not write so much detail, but it is a kindness that has gone too far.<BR>
ver 1.2<BR>
Added UTF-8 to avoid garbled Japanese characters after OCR processing.<BR>
ver 1.1<BR>
Added dark mode.<BR>
ver 1.0<BR>
The first version released. It has the ability to display the sum of the numbers read, extract strings, and output a comma-separated csv.<BR>
