#Methods

#Grayscale - Converts a color image into shades of gray.
#CLAHE - Improves local contrast so details and text are easier to see.
#Gaussian Blur - Smooths the image to reduce general/random noise.
#Bilateral Filter - Reduces noise while preserving important edges.
#Median Filter - Removes small dots and specks, especially salt-and-pepper noise.
#Adaptive Threshold - Converts an image to black and white based on local brightness.
#Canny - Detects edges and boundaries in an image.
#Morphological Opening - Removes small unwanted noise and tiny objects while preserving the main shapes of the image.


import cv2
import os  #helps Python find folders and files
import pytesseract  #used for OCR text recognition

#folder locations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  #where HW1.py is located
IMAGE_DIR = os.path.join(BASE_DIR, "images")  #original images folder
OUTPUT_DIR = os.path.join(BASE_DIR, "output")  #processed images folder
os.makedirs(OUTPUT_DIR, exist_ok=True)  #creates output folder if it does not exist

# ============================================================
# TESSERACT OCR
# ============================================================

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  #Tesseract installation path

# ============================================================
# OCR WORD SCORE FUNCTION
# ============================================================

def get_ocr_word_score(image, psm=6):

    #runs OCR and gets the detected words with their confidence scores
    data = pytesseract.image_to_data(image, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
    confidences = []
    words = []

    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        try:
            confidence = float(data["conf"][i])  #gets the confidence score for each word
        except:
            confidence = -1  #ignores invalid confidence values

        if word != "" and confidence >= 0:
            confidences.append(confidence)
            words.append(word)

    if len(confidences) == 0:
        return 0.0, 0, ""  #returns zero if no words are detected

    score = sum(confidences) / len(confidences)  #calculates average word confidence
    text = " ".join(words)  #combines detected words into one text
    return score, len(words), text

def get_best_ocr_result(image):

    #tries different Tesseract layouts because receipts can have different text arrangements
    psm_modes = [6, 4, 11]
    best_score = 0.0
    best_words = 0
    best_text = ""
    best_psm = 6

    for psm in psm_modes:
        score, words, text = get_ocr_word_score(image, psm)
        if words > 0 and score > best_score:
            best_score = score
            best_words = words
            best_text = text
            best_psm = psm
    return best_score, best_words, best_text, best_psm

# ============================================================
# 1. RECEIPT
# ============================================================

img = cv2.imread(os.path.join(IMAGE_DIR, "receipt_photo.jpg"))  #loads the receipt image

if img is None:
    print("ERROR: receipt_photo.jpg could not be loaded.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #converts receipt to grayscale

#save original
cv2.imwrite(os.path.join(OUTPUT_DIR, "receipt_before.png"), gray)

#lecture pipeline:
#Gray -> CLAHE -> Bilateral -> Adaptive Threshold -> Median

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)  #improves local contrast
den = cv2.bilateralFilter(clahe, 7, 50, 50)  #7 = filter size, 50 = color smoothing, 50 = spatial smoothing
bw = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 18)  #41 and 18 control the threshold
bw = cv2.medianBlur(bw, 3)  #removes small remaining noise

#small cleanup after filtering
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  #creates a small cleanup kernel
bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)  #removes tiny unwanted dots while keeping text

#save processed receipt
cv2.imwrite(os.path.join(OUTPUT_DIR, "receipt_clean.png"), bw)

# ------------------------------------------------------------
# RECEIPT - ZOOMED BEFORE / AFTER
# ------------------------------------------------------------

h, w = gray.shape  #gets image height and width
y1 = int(h * 0.25)  #Y start position
y2 = int(h * 0.50)  #Y end position
x1 = int(w * 0.25)  #X start position
x2 = int(w * 0.75)  #X end position

#y1 must be smaller than y2, and x1 must be smaller than x2
cv2.imwrite(os.path.join(OUTPUT_DIR, "receipt_before_crop.png"), gray[y1:y2, x1:x2])
cv2.imwrite(os.path.join(OUTPUT_DIR, "receipt_after_crop.png"), bw[y1:y2, x1:x2])

# ------------------------------------------------------------
# RECEIPT - INTENTIONALLY WRONG ORDER
# ------------------------------------------------------------

#correct order:
#CLAHE → Bilateral → Adaptive Threshold → Median

#wrong order:
#Gray → Adaptive Threshold → CLAHE → Bilateral

wrong = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 18)  #thresholding is done first
wrong = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(wrong)  #CLAHE is applied after thresholding
wrong = cv2.bilateralFilter(wrong, 7, 50, 50)  #bilateral filter is applied last
cv2.imwrite(os.path.join(OUTPUT_DIR, "receipt_wrong_order.png"), wrong)

# ============================================================
# RECEIPT OCR WORD SCORES
# ============================================================

before_score, before_words, before_text, before_psm = get_best_ocr_result(gray)  #OCR result before preprocessing
after_score, after_words, after_text, after_psm = get_best_ocr_result(bw)  #OCR result after preprocessing
wrong_score, wrong_words, wrong_text, wrong_psm = get_best_ocr_result(wrong)  #OCR result using wrong order
improvement = after_score - before_score  #difference between after and before scores

print("RECEIPT OCR WORD SCORES")
print("=======================")
print(f"Before preprocessing: {before_score:.2f}% ({before_words} words detected, PSM {before_psm})")
print(f"After preprocessing: {after_score:.2f}% ({after_words} words detected, PSM {after_psm})")
print(f"Wrong-order pipeline: {wrong_score:.2f}% ({wrong_words} words detected, PSM {wrong_psm})")
print(f"Improvement after preprocessing: {improvement:+.2f} percentage points")

# ============================================================
# SAVE OCR RESULTS
# ============================================================

ocr_result_path = os.path.join(OUTPUT_DIR, "receipt_ocr_results.txt")  #file for saving OCR results

with open(ocr_result_path, "w", encoding="utf-8") as file:
    file.write("RECEIPT OCR WORD SCORES\n")
    file.write("=======================\n\n")
    file.write(f"Before preprocessing: {before_score:.2f}% ({before_words} words detected, PSM {before_psm})\n")
    file.write(f"After preprocessing: {after_score:.2f}% ({after_words} words detected, PSM {after_psm})\n")
    file.write(f"Wrong-order pipeline: {wrong_score:.2f}% ({wrong_words} words detected, PSM {wrong_psm})\n")
    file.write(f"Improvement after preprocessing: {improvement:+.2f} percentage points\n\n")
    file.write("OCR TEXT - BEFORE PREPROCESSING\n")
    file.write("================================\n")
    file.write(before_text)
    file.write("\n\nOCR TEXT - AFTER PREPROCESSING\n")
    file.write("===============================\n")
    file.write(after_text)
    file.write("\n\nOCR TEXT - WRONG ORDER\n")
    file.write("=======================\n")
    file.write(wrong_text)

# ============================================================
# 2. GAUSSIAN NOISE
# ============================================================

img = cv2.imread(os.path.join(IMAGE_DIR, "noise_gauss.jpg"))  #loads Gaussian noise image

if img is not None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #converts image to grayscale
    
    #save original
    cv2.imwrite(os.path.join(OUTPUT_DIR, "gaussian_before.png"), gray)

    #Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  #change (5, 5) to see how the blur affects the result

    #Canny edge detection
    edges = cv2.Canny(blur, 50, 150)  #detects edges after removing noise

    #save result
    cv2.imwrite(os.path.join(OUTPUT_DIR, "noise_gauss_edges.png"), edges)

    # --------------------------------------------------------
    # GAUSSIAN - ZOOMED BEFORE / AFTER
    # --------------------------------------------------------

    h, w = gray.shape
    y1 = int(h * 0.25)
    y2 = int(h * 0.50)
    x1 = int(w * 0.25)
    x2 = int(w * 0.75)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "gaussian_before_crop.png"), gray[y1:y2, x1:x2])

    #AFTER = Gaussian Blur, not Canny
    cv2.imwrite(os.path.join(OUTPUT_DIR, "gaussian_after_crop.png"), blur[y1:y2, x1:x2])

# ============================================================
# 3. SALT-AND-PEPPER NOISE
# ============================================================

img = cv2.imread(os.path.join(IMAGE_DIR, "noise_sp.jpg"))  #loads salt-and-pepper noise image

if img is not None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #converts image to grayscale

    #save original
    cv2.imwrite(os.path.join(OUTPUT_DIR, "sp_before.png"), gray)

    #Median Filter
    blur = cv2.medianBlur(gray, 5)  #change 5 to see how the filter affects the result

    #Canny edge detection
    edges = cv2.Canny(blur, 50, 150)  #detects edges after removing noise

    #save result
    cv2.imwrite(os.path.join(OUTPUT_DIR, "noise_sp_edges.png"), edges)

    # --------------------------------------------------------
    # SALT-AND-PEPPER - ZOOMED BEFORE / AFTER
    # --------------------------------------------------------

    h, w = gray.shape
    y1 = int(h * 0.25)
    y2 = int(h * 0.50)
    x1 = int(w * 0.25)
    x2 = int(w * 0.75)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "sp_before_crop.png"), gray[y1:y2, x1:x2])

    #AFTER = Median Filter, not Canny
    cv2.imwrite(os.path.join(OUTPUT_DIR, "sp_after_crop.png"), blur[y1:y2, x1:x2])

# ============================================================
# 4. SUDOKU
# ============================================================

img = cv2.imread(os.path.join(IMAGE_DIR, "sudoku.jpg"))  #loads Sudoku image

if img is not None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #converts image to grayscale

    #save original
    cv2.imwrite(os.path.join(OUTPUT_DIR, "sudoku_before.png"), gray)

    #Median Filter
    blur = cv2.medianBlur(gray, 5)  #change 5 to see how the filter affects the result

    #Adaptive Threshold
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)  #change 11 and 2 to see how the result changes

    #save result
    cv2.imwrite(os.path.join(OUTPUT_DIR, "sudoku_binary.png"), binary)

    # --------------------------------------------------------
    # SUDOKU - ZOOMED BEFORE / AFTER
    # --------------------------------------------------------

    h, w = gray.shape
    y1 = int(h * 0.25)
    y2 = int(h * 0.50)
    x1 = int(w * 0.25)
    x2 = int(w * 0.75)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "sudoku_before_crop.png"), gray[y1:y2, x1:x2])
    cv2.imwrite(os.path.join(OUTPUT_DIR, "sudoku_after_crop.png"), binary[y1:y2, x1:x2])