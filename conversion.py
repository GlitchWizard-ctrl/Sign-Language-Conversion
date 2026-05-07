# ============================================================
# conversion.py — Sign ↔ Text ↔ Audio Pipeline (English Only)
# ============================================================

import os
import pickle
import numpy as np
import cv2

try:
    import speech_recognition as sr
except ImportError:
    raise ImportError("Run: pip install SpeechRecognition")

try:
    import pyttsx3
except ImportError:
    raise ImportError("Run: pip install pyttsx3")

# ============================================================
# CONFIG
# ============================================================
BASE_PATH    = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_PATH, "datasets")
MODEL_PATH   = os.path.join(BASE_PATH, "models")
DATA_PATH    = r"C:\Users\asus\Downloads\data"

# ============================================================
# LOAD LABEL ENCODER
# ============================================================
def load_label_encoder():
    path = os.path.join(DATASET_PATH, "label_encoder.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError("Run clean.py first to generate label_encoder.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

# ============================================================
# LOAD MODEL + SCALER
# ============================================================
def load_model():
    best_path   = os.path.join(MODEL_PATH, "best_model.pkl")
    svm_path    = os.path.join(MODEL_PATH, "sign_svm.pkl")
    scaler_path = os.path.join(MODEL_PATH, "hog_scaler.pkl")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Run train.py first to generate models.")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    if os.path.exists(best_path):
        with open(best_path, "rb") as f:
            data  = pickle.load(f)
            model = data["model"]
            name  = data["name"]
        print(f"  ✓ Loaded best model: {name}")
    elif os.path.exists(svm_path):
        with open(svm_path, "rb") as f:
            model = pickle.load(f)
        print("  ✓ Loaded SVM model")
    else:
        raise FileNotFoundError("No model found. Run train.py first.")

    return model, scaler

# ============================================================
# HOG FEATURE EXTRACTION
# Must match train.py exactly
# ============================================================
def extract_hog(img_rgb_224):
    hog = cv2.HOGDescriptor(
        _winSize    =(64, 64),
        _blockSize  =(16, 16),
        _blockStride=(8,  8),
        _cellSize   =(8,  8),
        _nbins      =9
    )
    img_uint8 = (img_rgb_224 * 255).astype(np.uint8)
    img_small = cv2.resize(img_uint8, (64, 64))
    img_gray  = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)
    feat      = hog.compute(img_gray).flatten()
    return feat

# ============================================================
# PREDICT SIGN FROM IMAGE
# ============================================================
def predict_sign(img_rgb_224, model, scaler, le):
    feat        = extract_hog(img_rgb_224).reshape(1, -1)
    feat_scaled = scaler.transform(feat)

    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(feat_scaled)[0]
        idx        = np.argmax(proba)
        confidence = proba[idx] * 100
    else:
        idx        = model.predict(feat_scaled)[0]
        confidence = None

    gloss = le.inverse_transform([idx])[0].upper()
    return gloss, confidence

# ============================================================
# SHOW SIGN IMAGE for a single gloss
# ============================================================
def show_sign_image(gloss, data_path):
    """
    Finds the folder matching the gloss label and
    displays a sample sign image for 2 seconds.
    """
    if not os.path.exists(data_path):
        print(f"  ✗ Data path not found: {data_path}")
        return

    for folder in os.listdir(data_path):
        if folder.upper() == gloss.upper():
            folder_path = os.path.join(data_path, folder)
            images      = sorted(os.listdir(folder_path))
            if images:
                img_path = os.path.join(folder_path, images[0])
                img      = cv2.imread(img_path)
                if img is not None:
                    img_resized = cv2.resize(img, (400, 400))

                    # Add label text on image
                    cv2.putText(
                        img_resized, gloss,
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3
                    )

                    cv2.imshow(f"ISL Sign: {gloss}", img_resized)
                    cv2.waitKey(8000)  # show 2 seconds per sign
                    cv2.destroyAllWindows()
                    return

    print(f"  ✗ No image found for: {gloss}")


# ============================================================
# SHOW FULL SENTENCE as sign images one by one
# ============================================================
def show_sentence_signs(gloss_sequence, data_path):
    """
    Shows sign images one by one for each gloss in sequence.
    Fingerspells letter by letter for unknown words.
    """
    print(f"\n  Displaying {len(gloss_sequence)} sign(s)...")
    print("  (Each sign shows for 2 seconds — press any key to skip)")

    for gloss in gloss_sequence:
        if gloss.startswith("[SPELL:"):
            # Fingerspell each letter
            word = gloss[7:-1]
            print(f"  Fingerspelling: {word}")
            for letter in word:
                print(f"    → Letter: {letter}")
                show_sign_image(letter, data_path)
        else:
            print(f"  → Sign: {gloss}")
            show_sign_image(gloss, data_path)

# ============================================================
# TRIE — fast gloss lookup
# ============================================================
class TrieNode:
    def __init__(self):
        self.children = {}
        self.gloss    = None

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, gloss):
        node = self.root
        for ch in word.lower():
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.gloss = gloss

    def search(self, word):
        node = self.root
        for ch in word.lower():
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node.gloss

def build_sign_database(le):
    trie = Trie()

    for cls in le.classes_:
        trie.insert(cls.lower(), cls.upper())

    extra_mappings = {
        "hello"   : "HELLO",
        "hi"      : "HELLO",
        "bye"     : "BYE",
        "goodbye" : "BYE",
        "yes"     : "YES",
        "no"      : "NO",
        "please"  : "PLEASE",
        "sorry"   : "SORRY",
        "thank"   : "THANK_YOU",
        "thanks"  : "THANK_YOU",
        "help"    : "HELP",
        "good"    : "GOOD",
        "bad"     : "BAD",
        "name"    : "NAME",
        "what"    : "WHAT",
        "where"   : "WHERE",
        "who"     : "WHO",
        "when"    : "WHEN",
        "how"     : "HOW",
        "i"       : "I",
        "you"     : "YOU",
        "we"      : "WE",
        "they"    : "THEY",
        "want"    : "WANT",
        "need"    : "NEED",
        "eat"     : "EAT",
        "drink"   : "DRINK",
        "go"      : "GO",
        "come"    : "COME",
        "stop"    : "STOP",
        "wait"    : "WAIT",
        "water"   : "WATER",
        "food"    : "FOOD",
        "home"    : "HOME",
        "school"  : "SCHOOL",
        "work"    : "WORK",
        "love"    : "LOVE",
        "happy"   : "HAPPY",
        "sad"     : "SAD",
        "morning" : "MORNING",
        "night"   : "NIGHT",
        "today"   : "TODAY",
        "tomorrow": "TOMORROW",
        "friend"  : "FRIEND",
        "family"  : "FAMILY",
        "mother"  : "MOTHER",
        "father"  : "FATHER",
        "brother" : "BROTHER",
        "sister"  : "SISTER",
    }

    for word, gloss in extra_mappings.items():
        trie.insert(word, gloss)

    return trie

# ============================================================
# NLP — Text → ISL Gloss sequence
# ISL drops articles/prepositions, follows SOV order
# ============================================================
STOP_WORDS = {"a", "an", "the", "is", "are", "was", "were",
              "be", "been", "being", "of", "in", "on", "at",
              "to", "for", "with", "by", "from", "and", "or"}

def text_to_gloss(sentence, trie):
    words          = sentence.lower().strip().split()
    gloss_sequence = []

    for word in words:
        clean = word.strip(".,!?;:'\"")
        if clean in STOP_WORDS:
            continue
        gloss = trie.search(clean)
        if gloss:
            gloss_sequence.append(gloss)
        else:
            gloss_sequence.append(f"[SPELL:{clean.upper()}]")

    return gloss_sequence

def gloss_to_sentence(gloss_list):
    cleaned  = [g for g in gloss_list if not g.startswith("[SPELL:")]
    spelled  = [g[7:-1] for g in gloss_list if g.startswith("[SPELL:")]
    sentence = " ".join(cleaned).lower().capitalize()
    if spelled:
        sentence += f" (fingerspelled: {', '.join(spelled)})"
    return sentence

# ============================================================
# SPEECH TO TEXT — supports .wav and .m4a (no PyAudio needed)
# ============================================================
def speech_to_text():
    audio_path = input("  Enter path to audio file (.wav or .m4a): ").strip()

    if not os.path.exists(audio_path):
        print("  ✗ File not found")
        return None

    # Convert .m4a to .wav if needed
    if audio_path.lower().endswith(".m4a"):
        try:
            from pydub import AudioSegment
            print("  Converting .m4a → .wav...")
            wav_path = audio_path.replace(".m4a", "_converted.wav")
            AudioSegment.from_file(audio_path, format="m4a").export(wav_path, format="wav")
            audio_path = wav_path
            print("  ✓ Conversion done")
        except ImportError:
            raise ImportError("Run: pip install pydub")
        except Exception as e:
            print(f"  ✗ Conversion failed: {e}")
            print("  Install ffmpeg: https://ffmpeg.org/download.html")
            return None

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language="en-US")
            print(f"  ✓ Heard: '{text}'")
            return text
        except sr.UnknownValueError:
            print("  ✗ Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"  ✗ Speech API error: {e}")
            return None

# ============================================================
# TEXT TO SPEECH — offline
# ============================================================
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.setProperty("rate",   150)
    engine.setProperty("volume", 1.0)
    engine.say(text)
    engine.runAndWait()

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    print("\n" + "="*55)
    print("   ISL Sign ↔ Text ↔ Audio Pipeline")
    print("="*55)

    # Load everything
    le            = load_label_encoder()
    model, scaler = load_model()
    trie          = build_sign_database(le)
    print(f"  ✓ Sign database ready ({len(le.classes_)} classes)\n")

    print("  Choose mode:")
    print("  [1] Text  → Sign Images  (Text-to-Sign)")
    print("  [2] Audio → Sign Images  (Speech-to-Sign)")
    print("  [3] Image → Text + Audio (Sign-to-Text-to-Speech)")
    print("  [4] Run all demos")

    choice = input("\n  Enter 1 / 2 / 3 / 4: ").strip()

    # ----------------------------------------------------------
    # MODE 1: Text → Gloss + Sign Images
    # ----------------------------------------------------------
    if choice in ("1", "4"):
        print("\n--- MODE 1: Text → ISL Sign Images ---")
        sentence  = input("  Enter English sentence: ").strip()
        gloss_seq = text_to_gloss(sentence, trie)
        print(f"  ISL Gloss Sequence : {' | '.join(gloss_seq)}")

        # Show sign images one by one
        show_sentence_signs(gloss_seq, DATA_PATH)

        speak = input("\n  Speak sentence aloud? (y/n): ").strip().lower()
        if speak == "y":
            text_to_speech(sentence)

    # ----------------------------------------------------------
    # MODE 2: Speech → Gloss + Sign Images
    # ----------------------------------------------------------
    if choice in ("2", "4"):
        print("\n--- MODE 2: Speech → ISL Sign Images ---")
        spoken = speech_to_text()
        if spoken:
            gloss_seq = text_to_gloss(spoken, trie)
            print(f"  ISL Gloss Sequence : {' | '.join(gloss_seq)}")

            # Show sign images one by one
            show_sentence_signs(gloss_seq, DATA_PATH)

            speak = input("\n  Speak sentence aloud? (y/n): ").strip().lower()
            if speak == "y":
                text_to_speech(spoken)

    # ----------------------------------------------------------
    # MODE 3: Sign Image → Text → Speech
    # ----------------------------------------------------------
    if choice in ("3", "4"):
        print("\n--- MODE 3: Sign Image → Text → Speech ---")
        img_path = input("  Enter path to sign image: ").strip()

        if not os.path.exists(img_path):
            print("  ✗ Image not found")
        else:
            img_bgr     = cv2.imread(img_path)
            img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            img_norm    = img_resized.astype(np.float32) / 255.0

            gloss, confidence = predict_sign(img_norm, model, scaler, le)
            sentence          = gloss_to_sentence([gloss])

            print(f"  Detected Gloss  : {gloss}")
            if confidence:
                print(f"  Confidence      : {confidence:.1f}%")
            print(f"  English Sentence: {sentence}")

            speak = input("  Speak it aloud? (y/n): ").strip().lower()
            if speak == "y":
                text_to_speech(sentence)

    print("\n✓ Pipeline complete.")