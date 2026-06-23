from flask import Flask, render_template_string, request, jsonify, send_file, Response
from datetime import datetime
from pathlib import Path
import random
import threading
import time
import socket
import os
import numpy as np
import requests

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    TENSORFLOW_AVAILABLE = True
except Exception as e:
    print("⚠ TensorFlow non disponible :", e)
    tf = None
    load_model = None
    image = None
    TENSORFLOW_AVAILABLE = False

from werkzeug.utils import secure_filename

# ============================================================
# Flask / paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)

UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

LOGO_FILE = BASE_DIR / "crown_logo.png"
IMAGE_URL = os.getenv(
    "DASHBOARD_IMAGE_URL",
    "https://images.unsplash.com/photo-1581092335397-9583eb92d232?auto=format&fit=crop&w=1200&q=80"
)

# ============================================================
# Phone camera IP Webcam
# ============================================================
PHONE_CAMERA_BASE_URL = "http://192.168.1.7:8082"
PHONE_CAMERA_SHOT_URL = PHONE_CAMERA_BASE_URL + "/shot.jpg"
PHONE_CAMERA_VIDEO_URL = PHONE_CAMERA_BASE_URL + "/video"


# ============================================================
# Raspberry / socket config
# ============================================================
server_ip = "192.168.48.238"
server_port_socket = 6000

# ============================================================
# AI models config
# ============================================================
BRAND_MODEL_PATH = BASE_DIR / "models" / "crown_clean_final.keras"
DEFECT_MODEL_PATH = BASE_DIR / "models" / "final_defect_classifier.keras"

BRAND_CLASSES = [
    "amestel beer", "apla", "arwa ananas", "arwa mandarine",
    "boga lemon", "celtia", "coca cola", "fanta",
    "hamoud", "selcto", "slim", "sprite", "viva citron"
]

# IMPORTANT: must match Colab train_generator.class_indices order
DEFECT_CLASSES = [
    "good",
    "pb color_contamination",
    "pb missing_paint"
]

BRAND_IMG_SIZE = (224, 224)
DEFECT_IMG_SIZE = (300, 300)

# Change this if your Colab preprocessing was different:
# "none", "rescale", "efficientnet", "mobilenetv2"
BRAND_PREPROCESS_MODE = "none"
DEFECT_PREPROCESS_MODE = "efficientnet"

# Decision thresholds
GOOD_DOMINANCE = 12.0
DEFECT_DOMINANCE = 10.0
SINGLE_GOOD_MIN = 70.0
SINGLE_DEFECT_MIN = 65.0
AVG_GOOD_MIN = 70.0
AVG_DEFECT_MIN = 60.0
MAX_DEFECT_MIN = 75.0
INSPECTION_360_FRAMES = 8

brand_model = None
defect_model = None

# ============================================================
# State
# ============================================================
BRANDS = [
    "Amestel Beer", "Apla", "Arwa Ananas", "Arwa Mandarine", "Boga Lemon",
    "Celtia", "Coca-Cola", "Fanta", "Hamoud", "Selecto", "Slim", "Sprite", "Viva Citron"
]

current_state = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "image_url": IMAGE_URL,
    "current_brand": "Coca-Cola",
    "brand_confidence": 99.4,
    "quality_status": "GOOD",
    "defect_type": "None",
    "defect_confidence": 0.0,
    "patchcore_result": "Normal",
    "efficientnet_result": "Good",
    "total_inspected": 12540,
    "good_cans": 12103,
    "defective_cans": 437,
    "accepted_percent": 96.52,
    "defect_rate": 3.48,
    "line_status": "Production Running",
    "camera_status": "Phone Camera",
    "ai_status": "Models Loading",
    "raspberry_status": "Connected",
    "database_status": "Connected / Local mode",
}

historical_data = {
    "timestamps": [],
    "good_rates": [],
    "defect_rates": [],
    "history_rows": [],
}


def recalculate_percentages():
    total = max(1, int(current_state["total_inspected"]))
    current_state["accepted_percent"] = round((current_state["good_cans"] / total) * 100, 2)
    current_state["defect_rate"] = round((current_state["defective_cans"] / total) * 100, 2)


def add_history_row(brand, result):
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    historical_data["timestamps"].append(current_timestamp)
    historical_data["good_rates"].append(current_state["accepted_percent"])
    historical_data["defect_rates"].append(current_state["defect_rate"])
    historical_data["history_rows"].append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "brand": brand,
        "result": result,
    })

    max_points = 50
    for key in historical_data:
        historical_data[key] = historical_data[key][-max_points:]


# ============================================================
# Model loading / preprocessing
# ============================================================
def load_ai_models():
    global brand_model, defect_model

    if not TENSORFLOW_AVAILABLE:
        current_state["ai_status"] = "TensorFlow Missing"
        return

    try:
        if BRAND_MODEL_PATH.exists():
            brand_model = load_model(BRAND_MODEL_PATH)
            print("✅ Brand model loaded:", BRAND_MODEL_PATH)
        else:
            print("⚠ Brand model not found:", BRAND_MODEL_PATH)

        if DEFECT_MODEL_PATH.exists():
            defect_model = load_model(DEFECT_MODEL_PATH)
            print("✅ Defect model loaded:", DEFECT_MODEL_PATH)
        else:
            print("⚠ Defect model not found:", DEFECT_MODEL_PATH)

        current_state["ai_status"] = "Models Online" if brand_model is not None and defect_model is not None else "Model Missing"

    except Exception as e:
        current_state["ai_status"] = "Model Load Error"
        print("❌ AI models loading error:", e)


def prepare_image_for_model(img_path, target_size, preprocess_mode="none"):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0).astype("float32")

    if preprocess_mode == "rescale":
        img_array = img_array / 255.0
    elif preprocess_mode == "efficientnet" and tf is not None:
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    elif preprocess_mode == "mobilenetv2" and tf is not None:
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    return img_array


def predict_brand_ai(img_path):
    if brand_model is None:
        raise RuntimeError("Brand model not loaded. Check models/crown_clean_final.keras")

    img_array = prepare_image_for_model(img_path, BRAND_IMG_SIZE, BRAND_PREPROCESS_MODE)
    preds = brand_model.predict(img_array, verbose=0)
    idx = int(np.argmax(preds[0]))
    return BRAND_CLASSES[idx], round(float(preds[0][idx] * 100), 2)


def normalize_defect_label(label):
    label = str(label).lower().strip()

    if label == "good":
        return "None", "Good", "GOOD", "Normal"
    if label == "suspect":
        return "À vérifier / 360°", "Suspect", "SUSPECT", "Need 360°"
    if "missing" in label:
        return "Missing Paint", "Missing Paint", "DEFECT", "Anomaly"
    if "contamination" in label or "color" in label:
        return "Color Contamination", "Color Contamination", "DEFECT", "Anomaly"

    return label, label, "DEFECT", "Anomaly"


def predict_defect_ai(img_path):
    if defect_model is None:
        raise RuntimeError("Defect model not loaded. Check models/final_defect_classifier.keras")

    img_array = prepare_image_for_model(img_path, DEFECT_IMG_SIZE, DEFECT_PREPROCESS_MODE)
    preds = defect_model.predict(img_array, verbose=0)[0]

    if len(preds) != len(DEFECT_CLASSES):
        raise RuntimeError(
            f"Nombre de sorties modèle ({len(preds)}) différent de DEFECT_CLASSES ({len(DEFECT_CLASSES)}). "
            "Vérifie train_generator.class_indices dans Colab."
        )

    scores = {cls: float(preds[i] * 100.0) for i, cls in enumerate(DEFECT_CLASSES)}
    good_score = scores.get("good", 0.0)
    color_score = scores.get("pb color_contamination", 0.0)
    missing_score = scores.get("pb missing_paint", 0.0)

    best_defect_score = max(color_score, missing_score)
    best_defect_label = "pb missing_paint" if missing_score >= color_score else "pb color_contamination"

    print("\n========== DEFECT PREDICTION ==========")
    print("Image:", img_path)
    print("DEFECT_CLASSES:", DEFECT_CLASSES)
    print(f"good={good_score:.2f}% | contamination={color_score:.2f}% | missing={missing_score:.2f}%")

    # Good if good clearly dominates
    if good_score >= SINGLE_GOOD_MIN and good_score >= best_defect_score + GOOD_DOMINANCE:
        raw_label = "good"
        confidence = round(good_score, 2)
        print("Decision: GOOD")
        print("=======================================\n")
        defect_type, efficientnet_result, quality_status, patchcore_result = normalize_defect_label(raw_label)
        return raw_label, defect_type, efficientnet_result, quality_status, patchcore_result, confidence, scores

    # Defect only if it clearly dominates good
    if best_defect_score >= SINGLE_DEFECT_MIN and best_defect_score >= good_score + DEFECT_DOMINANCE:
        raw_label = best_defect_label
        confidence = round(best_defect_score, 2)
        print("Decision: DEFECT", raw_label)
        print("=======================================\n")
        defect_type, efficientnet_result, quality_status, patchcore_result = normalize_defect_label(raw_label)
        return raw_label, defect_type, efficientnet_result, quality_status, patchcore_result, confidence, scores

    # Don't force contamination if classes are close
    raw_label = "suspect"
    confidence = round(max(good_score, best_defect_score), 2)
    print("Decision: SUSPECT, no clear dominance")
    print("=======================================\n")
    return raw_label, "À vérifier / 360°", "Suspect", "SUSPECT", "Need 360°", confidence, scores


def predict_full_image(img_path):
    brand, brand_conf = predict_brand_ai(str(img_path))
    raw_defect, defect_type, efficientnet_result, quality_status, patchcore_result, defect_conf, defect_scores = predict_defect_ai(str(img_path))
    relative_image = "/" + str(Path(img_path).relative_to(BASE_DIR)).replace("\\", "/")

    return {
        "brand": brand.title(),
        "brand_confidence": brand_conf,
        "raw_defect": raw_defect,
        "defect_type": defect_type,
        "defect_confidence": defect_conf,
        "defect_scores": defect_scores,
        "quality_status": quality_status,
        "patchcore_result": patchcore_result,
        "efficientnet_result": efficientnet_result,
        "image_path": relative_image,
    }


def aggregate_360_results(results):
    if not results:
        raise RuntimeError("Aucune frame 360° reçue")

    best_brand = max(results, key=lambda r: r["brand_confidence"])

    good_scores, color_scores, missing_scores = [], [], []
    for r in results:
        scores = r.get("defect_scores", {}) or {}
        good_scores.append(float(scores.get("good", 0.0)))
        color_scores.append(float(scores.get("pb color_contamination", 0.0)))
        missing_scores.append(float(scores.get("pb missing_paint", 0.0)))

    avg_good = sum(good_scores) / len(good_scores)
    avg_color = sum(color_scores) / len(color_scores)
    avg_missing = sum(missing_scores) / len(missing_scores)

    max_good = max(good_scores)
    max_color = max(color_scores)
    max_missing = max(missing_scores)

    avg_best_defect = max(avg_color, avg_missing)

    if avg_missing >= avg_color:
        best_defect_label = "pb missing_paint"
        best_defect_type = "Missing Paint"
        best_defect_conf = round(max_missing, 2)
        best_frame = max(results, key=lambda r: float((r.get("defect_scores", {}) or {}).get("pb missing_paint", 0.0)))
    else:
        best_defect_label = "pb color_contamination"
        best_defect_type = "Color Contamination"
        best_defect_conf = round(max_color, 2)
        best_frame = max(results, key=lambda r: float((r.get("defect_scores", {}) or {}).get("pb color_contamination", 0.0)))

    debug_scores = {
        "avg_good": round(avg_good, 2),
        "avg_color_contamination": round(avg_color, 2),
        "avg_missing_paint": round(avg_missing, 2),
        "max_good": round(max_good, 2),
        "max_color_contamination": round(max_color, 2),
        "max_missing_paint": round(max_missing, 2),
    }

    print("\n========== 360 AGGREGATION ==========")
    print(debug_scores)

    # GOOD priority if it dominates on average
    if avg_good >= AVG_GOOD_MIN and avg_good >= avg_best_defect + GOOD_DOMINANCE:
        best_good = max(results, key=lambda r: float((r.get("defect_scores", {}) or {}).get("good", 0.0)))
        print("Final decision: GOOD")
        print("=====================================\n")
        return {
            "brand": best_brand["brand"],
            "brand_confidence": best_brand["brand_confidence"],
            "raw_defect": "good",
            "defect_type": "None",
            "defect_confidence": round(avg_good, 2),
            "defect_scores": debug_scores,
            "quality_status": "GOOD",
            "patchcore_result": "Normal",
            "efficientnet_result": "Good",
            "image_path": best_good["image_path"],
            "frames_analyzed": len(results),
            "decision_rule": "GOOD confirmé: le score GOOD domine sur le tour 360°",
            "frames": results,
        }

    # DEFECT only if defect clearly dominates
    defect_avg_dominates = avg_best_defect >= AVG_DEFECT_MIN and avg_best_defect >= avg_good + DEFECT_DOMINANCE
    defect_max_dominates = best_defect_conf >= MAX_DEFECT_MIN and best_defect_conf >= max_good + 8.0

    if defect_avg_dominates or defect_max_dominates:
        print("Final decision: DEFECT", best_defect_type)
        print("=====================================\n")
        return {
            "brand": best_brand["brand"],
            "brand_confidence": best_brand["brand_confidence"],
            "raw_defect": best_defect_label,
            "defect_type": best_defect_type,
            "defect_confidence": best_defect_conf,
            "defect_scores": debug_scores,
            "quality_status": "DEFECT",
            "patchcore_result": "Anomaly",
            "efficientnet_result": best_defect_type,
            "image_path": best_frame["image_path"],
            "frames_analyzed": len(results),
            "decision_rule": "DEFECT confirmé: le défaut domine clairement GOOD après 360°",
            "frames": results,
        }

    print("Final decision: SUSPECT")
    print("=====================================\n")
    return {
        "brand": best_brand["brand"],
        "brand_confidence": best_brand["brand_confidence"],
        "raw_defect": "suspect",
        "defect_type": "À vérifier / refaire 360° plus proche",
        "defect_confidence": round(max(avg_good, avg_best_defect), 2),
        "defect_scores": debug_scores,
        "quality_status": "SUSPECT",
        "patchcore_result": "Need verification",
        "efficientnet_result": "Suspect",
        "image_path": best_frame["image_path"],
        "frames_analyzed": len(results),
        "decision_rule": "SUSPECT: aucun score ne domine clairement",
        "frames": results,
    }


def update_state_from_prediction(result, count_as_one=True):
    current_state["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_state["image_url"] = result.get("image_path", current_state.get("image_url", IMAGE_URL))
    current_state["current_brand"] = result["brand"].title()
    current_state["brand_confidence"] = result["brand_confidence"]
    current_state["quality_status"] = result["quality_status"]
    current_state["defect_type"] = result["defect_type"]
    current_state["defect_confidence"] = result["defect_confidence"]
    current_state["patchcore_result"] = result["patchcore_result"]
    current_state["efficientnet_result"] = result["efficientnet_result"]
    current_state["ai_status"] = "Models Online"

    if count_as_one:
        current_state["total_inspected"] += 1
        if result["quality_status"] == "GOOD":
            current_state["good_cans"] += 1
        else:
            current_state["defective_cans"] += 1

    recalculate_percentages()
    add_history_row(
        current_state["current_brand"],
        "Good" if result["quality_status"] == "GOOD" else result["defect_type"]
    )


# ============================================================
# Phone camera helpers
# ============================================================
def capture_from_phone_camera():
    try:
        response = requests.get(PHONE_CAMERA_SHOT_URL, timeout=5)
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_path = UPLOAD_FOLDER / f"{timestamp}_phone_camera.jpg"
        with open(img_path, "wb") as f:
            f.write(response.content)
        return img_path
    except Exception as e:
        raise RuntimeError(f"Erreur caméra téléphone : {e}")


def send_to_server(command):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as clientsocket:
            clientsocket.settimeout(3)
            clientsocket.connect((server_ip, server_port_socket))
            clientsocket.sendall(command.encode())
    except Exception as e:
        print("Erreur d'envoi socket :", e)


def update_quality_data():
    defect_types = ["Color Contamination", "Missing Paint"]

    while True:
        current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        brand = random.choice(BRANDS)
        is_defect = random.choice([False, False, False, False, True])

        new_cans = random.randint(3, 10)
        current_state["total_inspected"] += new_cans

        if is_defect:
            defect_type = random.choice(defect_types)
            current_state["defective_cans"] += 1
            quality_status = "DEFECT"
            patchcore_result = "Anomaly"
            efficientnet_result = defect_type
        else:
            defect_type = "None"
            quality_status = "GOOD"
            patchcore_result = "Normal"
            efficientnet_result = "Good"

        current_state["good_cans"] = current_state["total_inspected"] - current_state["defective_cans"]
        current_state.update({
            "timestamp": current_timestamp,
            "image_url": IMAGE_URL,
            "current_brand": brand,
            "brand_confidence": round(random.uniform(97.0, 99.9), 1),
            "quality_status": quality_status,
            "defect_type": defect_type,
            "defect_confidence": round(random.uniform(91.0, 99.0), 1),
            "patchcore_result": patchcore_result,
            "efficientnet_result": efficientnet_result,
            "line_status": "Production Running",
            "camera_status": "Phone Camera",
            "ai_status": "Models Online",
            "raspberry_status": "Connected",
        })
        recalculate_percentages()
        add_history_row(brand, defect_type if is_defect else "Good")
        time.sleep(5)


SIMULATION_MODE = False
load_ai_models()
if SIMULATION_MODE:
    threading.Thread(target=update_quality_data, daemon=True).start()


# ============================================================
# Routes
# ============================================================
@app.route("/logo")
def logo():
    if LOGO_FILE.exists():
        return send_file(LOGO_FILE)
    return "", 404


@app.route("/phone-camera-shot")
def phone_camera_shot():
    try:
        response = requests.get(PHONE_CAMERA_SHOT_URL, timeout=5)
        if response.status_code != 200:
            return "", response.status_code
        return response.content, 200, {
            "Content-Type": "image/jpeg",
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        }
    except Exception as e:
        # Return a small transparent response if camera is unavailable
        print("Erreur phone-camera-shot:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/")
def dashboard():
    brand_stats = [
        {"brand": "Coca-Cola", "produced": 2500, "defects": 35},
        {"brand": "Fanta", "produced": 1800, "defects": 42},
        {"brand": "Hamoud", "produced": 1200, "defects": 18},
        {"brand": "Sprite", "produced": 900, "defects": 11},
    ]

    return render_template_string(DASHBOARD_HTML, current=current_state, history=historical_data, history_rows=historical_data["history_rows"], brand_stats=brand_stats)


@app.route("/predict-image", methods=["POST"])
def predict_image():
    if brand_model is None or defect_model is None:
        return jsonify({"error": "Modèles IA non chargés. Vérifie le dossier models/."}), 500

    try:
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            img_path = UPLOAD_FOLDER / f"{timestamp}_{filename}"
            file.save(img_path)
        else:
            img_path = capture_from_phone_camera()

        result = predict_full_image(img_path)

        # One image is indicative only
        if result.get("quality_status") == "GOOD":
            result["quality_status"] = "INSPECT 360°"
            result["defect_type"] = "Appuyer sur Vérifier défectueuse ou pas"
            result["efficientnet_result"] = "En attente inspection 360°"
            result["patchcore_result"] = "Need 360°"
            result["raw_defect"] = "need_360"

        result["decision_rule"] = "Image simple: résultat indicatif. Utiliser Vérifier défectueuse ou pas."
        update_state_from_prediction(result, count_as_one=False)

    except Exception as e:
        return jsonify({"error": f"Erreur prédiction IA : {e}"}), 500

    result["current"] = current_state
    return jsonify(result)


@app.route("/predict-360", methods=["POST"])
def predict_360():
    if brand_model is None or defect_model is None:
        return jsonify({"error": "Modèles IA non chargés. Vérifie le dossier models/."}), 500

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    try:
        files = request.files.getlist("frames")

        if files:
            for i, file in enumerate(files, start=1):
                filename = secure_filename(file.filename or f"view_{i}.jpg")
                img_path = UPLOAD_FOLDER / f"{timestamp}_360_{i}_{filename}"
                file.save(img_path)
                results.append(predict_full_image(img_path))
        else:
            # Phone camera mode: capture 8 frames from IP Webcam
            for i in range(1, INSPECTION_360_FRAMES + 1):
                img_path = capture_from_phone_camera()
                renamed_path = UPLOAD_FOLDER / f"{timestamp}_360_{i}_phone.jpg"
                try:
                    Path(img_path).rename(renamed_path)
                    img_path = renamed_path
                except Exception:
                    pass
                results.append(predict_full_image(img_path))
                time.sleep(0.45)

        final_result = aggregate_360_results(results)
        update_state_from_prediction(final_result, count_as_one=True)

    except Exception as e:
        return jsonify({"error": f"Erreur prédiction 360° : {e}"}), 500

    final_result["current"] = current_state
    return jsonify(final_result)


@app.route("/quality-data")
def quality_data():
    return jsonify({"current": current_state, "history": historical_data})


@app.route("/chatbot")
def chatbot_page():
    return render_template_string(CHATBOT_HTML)


def _clean_question(text):
    text = str(text or "").lower().strip()
    accents = {"é": "e", "è": "e", "ê": "e", "ë": "e", "à": "a", "â": "a", "î": "i", "ï": "i", "ô": "o", "ù": "u", "û": "u", "ç": "c"}
    for old, new_char in accents.items():
        text = text.replace(old, new_char)
    return text


def _last_detection_answer():
    brand = current_state.get("current_brand", "Non disponible")
    status = current_state.get("quality_status", "Non disponible")
    defect = current_state.get("defect_type", "None")
    efficientnet = current_state.get("efficientnet_result", "Non disponible")
    patchcore = current_state.get("patchcore_result", "Non disponible")

    if status == "INSPECT 360°":
        return f"Dernière marque détectée : {brand}.\n\nLa canette n'a pas encore été validée. Il faut appuyer sur 'Vérifier défectueuse ou pas' pour lancer l'inspection 360°."
    if status == "GOOD":
        return f"Dernière marque détectée : {brand}.\nStatut qualité : GOOD.\n\nAucun défaut visible n'a été confirmé."
    if status == "DEFECT":
        return f"Dernière marque détectée : {brand}.\nStatut qualité : DEFECT.\nDéfaut détecté : {defect}.\n\nEfficientNetB0 : {efficientnet}.\nPatchCore : {patchcore}."
    if status == "SUSPECT":
        return f"Dernière marque détectée : {brand}.\nStatut qualité : SUSPECT.\n\nRefais une inspection 360° avec la canette bien placée dans le rectangle vert."
    return f"Dernière marque détectée : {brand}.\nStatut qualité actuel : {status}.\nDéfaut : {defect}."


def _stats_answer():
    return (
        f"Statistiques actuelles du dashboard :\n\n"
        f"Total inspecté : {current_state.get('total_inspected')} canettes.\n"
        f"Canettes acceptées : {current_state.get('good_cans')}.\n"
        f"Canettes rejetées ou suspectes : {current_state.get('defective_cans')}.\n"
        f"Taux d'acceptation : {current_state.get('accepted_percent')}%.\n"
        f"Taux de défaut : {current_state.get('defect_rate')}%."
    )


LOCAL_KNOWLEDGE = [
    (("inspection 360", "360", "verifier defectueuse", "verifier defaut"),
     "L'inspection 360° permet d'analyser toute la surface de la canette. Une seule image frontale peut manquer un défaut situé sur l'arrière ou sur le côté."),
    (("missing paint", "manque de peinture", "peinture manquante", "absence de peinture"),
     "Missing Paint signifie qu'une zone de la canette n'a pas reçu correctement la peinture ou le décor."),
    (("color contamination", "contamination", "tache", "couleur parasite"),
     "Color Contamination correspond à une couleur parasite ou une tache inattendue sur la canette."),
    (("patchcore", "anomalie", "anomaly"),
     "PatchCore est utilisé comme module de détection d'anomalies."),
    (("efficientnet", "efficientnetb0", "classification defaut", "classifier"),
     "EfficientNetB0 sert à classifier Good, Missing Paint et Color Contamination."),
    (("marque", "brand", "categorie", "sprite", "coca", "slim", "hamoud", "fanta", "selecto"),
     "Le premier modèle IA détecte la marque ou la catégorie de la canette."),
    (("raspberry", "camera", "deploiement", "prototype"),
     "Le système peut être déployé sur Raspberry Pi avec une caméra."),
]


@app.route("/chatbot-api", methods=["POST"])
def chatbot_api():
    data = request.get_json(silent=True) or {}
    user_message = str(data.get("message", "")).strip()
    if not user_message:
        return jsonify({"reply": "Veuillez écrire une question."})

    q = _clean_question(user_message)

    if any(word in q for word in ["dernier", "resultat", "detection", "detecte", "explique"]):
        return jsonify({"reply": _last_detection_answer()})

    if any(word in q for word in ["statistique", "stats", "combien", "total", "taux", "production"]):
        return jsonify({"reply": _stats_answer()})

    if any(word in q for word in ["bonjour", "salut", "hello", "bonsoir"]):
        return jsonify({"reply": "Bonjour, je suis l'assistant IA local du dashboard Crown Maghreb."})

    for keywords, answer in LOCAL_KNOWLEDGE:
        if any(key in q for key in keywords):
            return jsonify({"reply": answer})

    return jsonify({"reply": "Je peux répondre aux questions sur la marque, l'inspection 360°, Missing Paint, Color Contamination, PatchCore, EfficientNetB0 et les statistiques."})


@app.route("/control")
def control_panel():
    recalculate_percentages()
    batch_id = f"BATCH-{current_state['current_brand'].upper().replace(' ', '-')}-2026-001"
    return render_template_string(CONTROL_HTML, current=current_state, batch_id=batch_id)


@app.route("/action", methods=["POST"])
def handle_action():
    data = request.get_json(silent=True) or {}
    action = data.get("action", "")
    print("Action reçue :", action)

    current_state["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if action == "simulate_good":
        current_state["total_inspected"] += 1
        current_state["good_cans"] += 1
        current_state["quality_status"] = "GOOD"
        current_state["defect_type"] = "None"
        current_state["defect_confidence"] = round(random.uniform(96.0, 99.8), 1)
        current_state["patchcore_result"] = "Normal"
        current_state["efficientnet_result"] = "Good"

    elif action == "simulate_missing_paint":
        current_state["total_inspected"] += 1
        current_state["defective_cans"] += 1
        current_state["quality_status"] = "DEFECT"
        current_state["defect_type"] = "Missing Paint"
        current_state["defect_confidence"] = round(random.uniform(91.0, 99.0), 1)
        current_state["patchcore_result"] = "Anomaly"
        current_state["efficientnet_result"] = "Missing Paint"

    elif action == "simulate_contamination":
        current_state["total_inspected"] += 1
        current_state["defective_cans"] += 1
        current_state["quality_status"] = "DEFECT"
        current_state["defect_type"] = "Color Contamination"
        current_state["defect_confidence"] = round(random.uniform(91.0, 99.0), 1)
        current_state["patchcore_result"] = "Anomaly"
        current_state["efficientnet_result"] = "Color Contamination"

    elif action == "reset_counters":
        current_state["total_inspected"] = 0
        current_state["good_cans"] = 0
        current_state["defective_cans"] = 0
        current_state["quality_status"] = "GOOD"
        current_state["defect_type"] = "None"
        current_state["defect_confidence"] = 0.0
        current_state["patchcore_result"] = "Normal"
        current_state["efficientnet_result"] = "Good"
        historical_data["timestamps"].clear()
        historical_data["good_rates"].clear()
        historical_data["defect_rates"].clear()
        historical_data["history_rows"].clear()

    elif action == "start_inspection":
        current_state["line_status"] = "Production Running"

    elif action == "pause_inspection":
        current_state["line_status"] = "Production Paused"

    elif action == "stop_inspection":
        current_state["line_status"] = "Production Stopped"

    elif action in ["good_led_on", "defect_led_on", "buzzeron", "buzzeroff", "green_led", "red_led", "buzzer", "reject_can", "reload_models", "test_camera", "restart_raspberry"]:
        send_to_server(action)

    recalculate_percentages()
    add_history_row(current_state["current_brand"], "Good" if current_state["quality_status"] == "GOOD" else current_state["defect_type"])

    return jsonify({"status": "ok", "action": action, "current": current_state})


# ============================================================
# HTML templates
# ============================================================
DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CROWN MAGHREB - Deco Machine Quality Inspection</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
:root { --green:#007a3d; --deep:#174d45; --teal:#2f7468; --red:#c0392b; --orange:#d57435; --bg:#eef6f2; --panel:#ffffffee; --border:#c8ddd8; --text:#203b36; --muted:#6b7d78; }
body { margin:0; min-height:100vh; background:linear-gradient(135deg,#eef6f2 0%,#fbf9ee 50%,#e4f1ee 100%); color:var(--text); font-family:"Segoe UI",system-ui,sans-serif; }
.layout { display:grid; grid-template-columns:220px 1fr; gap:18px; padding:18px; }
.sidebar,.panel,.kpi,.header { background:var(--panel); border:1px solid var(--border); box-shadow:0 12px 28px rgba(28,92,83,.12); border-radius:18px; }
.sidebar { background:linear-gradient(180deg,#3f837a,#174d45); color:white; padding:16px; min-height:calc(100vh - 36px); }
.logo-box { background:white; border-radius:16px; padding:12px; text-align:center; margin-bottom:18px; }
.logo-box img { max-width:160px; width:100%; }
.nav-line { padding:11px 10px; border-radius:12px; margin-bottom:7px; color:#ffffffdd; }
.nav-line.active,.nav-line:hover { background:rgba(255,255,255,.18); color:white; }
.header { padding:18px 22px; margin-bottom:16px; display:flex; justify-content:space-between; align-items:center; gap:15px; }
.header h1 { font-size:1.4rem; font-weight:900; margin:0; color:var(--deep); }
.subtitle { color:var(--muted); font-size:.92rem; margin-top:4px; }
.status { display:flex; gap:9px; flex-wrap:wrap; justify-content:end; }
.status-pill { background:#f4fbf7; border:1px solid var(--border); border-radius:999px; padding:7px 10px; font-weight:700; font-size:.78rem; }
.dot { width:9px; height:9px; display:inline-block; border-radius:50%; background:#17a75a; margin-right:6px; box-shadow:0 0 0 4px rgba(23,167,90,.12); }
.kpi-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:16px; }
.kpi { padding:16px; min-height:118px; }
.kpi .icon { width:38px; height:38px; border-radius:11px; display:grid; place-items:center; background:#e6f4ef; color:var(--green); margin-bottom:8px; }
.kpi-value { font-size:1.85rem; line-height:1; font-weight:900; color:var(--teal); }
.kpi-label { color:var(--muted); font-size:.82rem; font-weight:700; margin-top:7px; }
.grid-main { display:grid; grid-template-columns:1.1fr .9fr .9fr; gap:16px; margin-bottom:16px; }
.grid-wide { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:16px; }
.panel { padding:16px; }
.panel-title { font-size:.98rem; font-weight:900; color:var(--deep); margin-bottom:12px; display:flex; align-items:center; justify-content:space-between; gap:12px; }
.camera-frame { height:285px; border-radius:16px; overflow:hidden; border:1px solid var(--border); background:#ddebe7; position:relative; }
.camera-frame img { width:100%; height:100%; object-fit:cover; display:block; }
.roi-guide { position:absolute; left:35%; top:7%; width:30%; height:86%; border:3px dashed #00a85a; border-radius:18px; box-shadow:0 0 0 9999px rgba(0,0,0,.10); pointer-events:none; }
.roi-text { position:absolute; left:35%; top:7%; transform:translateY(-110%); background:#ffffffee; border:1px solid var(--border); border-radius:999px; padding:5px 10px; font-weight:900; color:var(--deep); font-size:.78rem; pointer-events:none; }
.badge-live { position:absolute; left:12px; bottom:12px; background:#fffffff0; border:1px solid var(--border); border-radius:999px; padding:8px 12px; font-weight:900; color:var(--deep); }
.big-status { border-radius:18px; padding:22px; text-align:center; font-size:2.2rem; font-weight:950; letter-spacing:1px; }
.good { background:#e7f7ef; color:#08783f; border:1px solid #bfe7d0; }
.defect { background:#fff0e8; color:#b83222; border:1px solid #f0c5b4; }
.info-row { display:flex; justify-content:space-between; border-bottom:1px solid #e2eeeb; padding:9px 0; gap:15px; }
.info-row span:first-child { color:var(--muted); font-weight:700; }
.info-row strong { text-align:right; }
.table { --bs-table-bg:transparent; font-size:.86rem; }
.chart-box { height:260px; }
.btn-crown { background:linear-gradient(135deg,var(--green),#0b5f35); border:0; color:white; border-radius:12px; font-weight:800; padding:9px 12px; }
.btn-alert { background:linear-gradient(135deg,#d57435,#a44a23); border:0; color:white; border-radius:12px; font-weight:800; padding:9px 12px; }
@media(max-width:1200px){ .layout{grid-template-columns:1fr}.sidebar{min-height:auto}.kpi-grid{grid-template-columns:repeat(2,1fr)}.grid-main,.grid-wide{grid-template-columns:1fr} }
</style>
</head>
<body>
<div class="layout">
<aside class="sidebar">
    <div class="logo-box"><img src="/logo" alt="CROWN logo"></div>
    <div class="nav-line active"><i class="fa-solid fa-gauge-high me-2"></i>Overview</div>
    <div class="nav-line"><i class="fa-solid fa-camera me-2"></i>Real-time Inspection</div>
    <div class="nav-line"><i class="fa-solid fa-robot me-2"></i>AI Models</div>
    <div class="nav-line"><i class="fa-solid fa-industry me-2"></i>Production</div>
    <div class="nav-line"><i class="fa-solid fa-chart-pie me-2"></i>Quality Statistics</div>
    <div class="nav-line"><i class="fa-solid fa-clock-rotate-left me-2"></i>Detection History</div>
    <a href="/control" class="nav-line" style="display:block;color:white;text-decoration:none;"><i class="fa-solid fa-screwdriver-wrench me-2"></i>Machine Control</a>
    <a href="/chatbot" class="nav-line" style="display:block;color:white;text-decoration:none;"><i class="fa-solid fa-comments me-2"></i>Assistant IA</a>
    <div class="mt-4 p-3 rounded-3" style="background:rgba(255,255,255,.12)"><div style="font-size:.75rem;opacity:.75">Project</div><strong>Deco Machine Quality Inspection</strong><div style="font-size:.78rem;opacity:.8;margin-top:6px">Brand recognition + defect detection</div></div>
</aside>

<main>
<section class="header">
    <div>
        <h1>CROWN MAGHREB<br>DECO MACHINE QUALITY INSPECTION SYSTEM</h1>
        <div class="subtitle">AI-Based Brand Recognition & Defect Detection</div>
    </div>
    <div class="status">
        <div class="status-pill"><span class="dot"></span>Phone Camera</div>
        <div class="status-pill"><span class="dot"></span>AI Models Online</div>
        <div class="status-pill"><span class="dot"></span>Raspberry Pi Connected</div>
        <div class="status-pill"><span class="dot"></span>Production Running</div>
        <a href="/control" class="btn btn-crown btn-sm"><i class="fa-solid fa-sliders me-1"></i>Control Center</a>
    </div>
</section>

<section class="kpi-grid">
    <div class="kpi"><div class="icon"><i class="fa-solid fa-boxes-stacked"></i></div><div id="totalInspected" class="kpi-value">{{ current.total_inspected }}</div><div class="kpi-label">TOTAL INSPECTED</div></div>
    <div class="kpi"><div class="icon"><i class="fa-solid fa-circle-check"></i></div><div id="goodCans" class="kpi-value">{{ current.good_cans }}</div><div class="kpi-label">ACCEPTED / GOOD</div><small id="acceptedPercent">{{ current.accepted_percent }}%</small></div>
    <div class="kpi"><div class="icon" style="background:#fff0e8;color:var(--orange)"><i class="fa-solid fa-triangle-exclamation"></i></div><div id="defectiveCans" class="kpi-value" style="color:var(--orange)">{{ current.defective_cans }}</div><div class="kpi-label">REJECTED / DEFECT</div><small id="defectRate">{{ current.defect_rate }}%</small></div>
    <div class="kpi"><div class="icon"><i class="fa-solid fa-tag"></i></div><div id="currentBrand" class="kpi-value" style="font-size:1.35rem">{{ current.current_brand }}</div><div class="kpi-label">CURRENT PRODUCTION</div><small>Confidence: <span id="brandConfidence">{{ current.brand_confidence }}</span>%</small></div>
</section>

<section class="grid-main">
    <div class="panel">
        <div class="panel-title">
            <span><i class="fa-solid fa-video me-2"></i>Inspection en Temps Réel</span>
            <div class="d-flex gap-2 flex-wrap">
                <button class="btn btn-crown btn-sm" onclick="openCamera()"><i class="fa-solid fa-mobile-screen me-1"></i>Open Phone Camera</button>
                <button class="btn btn-alert btn-sm" onclick="captureAndPredict()"><i class="fa-solid fa-brain me-1"></i>Détecter marque</button>
                <button class="btn btn-crown btn-sm" onclick="capture360AndPredict()"><i class="fa-solid fa-arrows-rotate me-1"></i>Vérifier défectueuse ou pas</button>
                <button class="btn btn-outline-secondary btn-sm" onclick="toggleAutoInspection()"><i class="fa-solid fa-rotate me-1"></i><span id="autoBtnText">Auto OFF</span></button>
            </div>
        </div>
        <div class="camera-frame">
            <img src="/phone-camera-shot?t={{ current.timestamp }}" id="cameraImage" alt="Phone camera feed" onerror="this.src='{{ current.image_url }}'">
            <div class="roi-guide"></div>
            <div class="roi-text">PLACE CAN HERE</div>
            <div class="badge-live" id="cameraBadge">PHONE CAMERA - IP Webcam</div>
        </div>
        <div id="cameraAiResult" class="mt-3"></div>
    </div>

    <div class="panel">
        <div class="panel-title"><i class="fa-solid fa-medal me-2"></i>Production Actuelle</div>
        <div class="info-row"><span>Brand</span><strong id="brandBox">{{ current.current_brand }}</strong></div>
        <div class="info-row"><span>Detection Confidence</span><strong><span id="brandConfidenceBox">{{ current.brand_confidence }}</span>%</strong></div>
        <hr>
        <div class="panel-title">Quality Status</div>
        <div id="qualityStatusBox" class="big-status {{ 'good' if current.quality_status == 'GOOD' else 'defect' }}">{{ current.quality_status }}</div>
        <div class="info-row mt-3"><span>Defect Type</span><strong id="defectType">{{ current.defect_type }}</strong></div>
    </div>

    <div class="panel">
        <div class="panel-title"><i class="fa-solid fa-brain me-2"></i>AI Models Status</div>
        <table class="table align-middle">
            <thead><tr><th>Module</th><th>Résultat</th></tr></thead>
            <tbody>
                <tr><td>Brand Classifier</td><td id="brandClassifier">{{ current.current_brand }}</td></tr>
                <tr><td>PatchCore</td><td id="patchcoreResult">{{ current.patchcore_result }}</td></tr>
                <tr><td>EfficientNetB0</td><td id="efficientnetResult">{{ current.efficientnet_result }}</td></tr>
            </tbody>
        </table>
        <div class="panel-title mt-3"><i class="fa-solid fa-microchip me-2"></i>System Information</div>
        <div class="info-row"><span>Hardware</span><strong>Raspberry Pi 4</strong></div>
        <div class="info-row"><span>Camera Status</span><strong id="cameraStatus">{{ current.camera_status }}</strong></div>
        <div class="info-row"><span>AI Status</span><strong id="aiStatus">{{ current.ai_status }}</strong></div>
        <div class="info-row"><span>Database</span><strong>{{ current.database_status }}</strong></div>
        <div class="info-row"><span>Last Inspection</span><strong id="lastInspection">{{ current.timestamp }}</strong></div>
    </div>
</section>

<section class="grid-wide">
    <div class="panel"><div class="panel-title">Répartition des Défauts</div><div class="chart-box"><canvas id="defectChart"></canvas></div></div>
    <div class="panel"><div class="panel-title">Production Quality Trend</div><div class="chart-box"><canvas id="qualityChart"></canvas></div></div>
</section>

<section class="grid-wide">
    <div class="panel">
        <div class="panel-title">Répartition par Marque</div>
        <table class="table table-hover align-middle mb-0">
            <thead><tr><th>Brand</th><th>Produced</th><th>Defects</th><th>Defect %</th></tr></thead>
            <tbody>
            {% for row in brand_stats %}<tr><td>{{ row.brand }}</td><td>{{ row.produced }}</td><td>{{ row.defects }}</td><td>{{ ((row.defects / row.produced) * 100) | round(2) }}%</td></tr>{% endfor %}
            </tbody>
        </table>
    </div>
    <div class="panel">
        <div class="panel-title">Historique des Dernières Détections</div>
        <table class="table table-hover align-middle mb-0">
            <thead><tr><th>Time</th><th>Brand</th><th>Result</th></tr></thead>
            <tbody id="historyTable">
            {% for row in history_rows|reverse %}<tr><td>{{ row.time }}</td><td>{{ row.brand }}</td><td>{{ row.result }}</td></tr>{% endfor %}
            </tbody>
        </table>
    </div>
</section>

<section class="panel mb-3">
    <div class="panel-title"><span><i class="fa-solid fa-upload me-2"></i>Test Image avec les Modèles IA</span></div>
    <form id="aiUploadForm" class="d-flex gap-2 flex-wrap align-items-center">
        <input class="form-control" style="max-width:420px" type="file" id="aiImage" name="file" accept="image/*">
        <button class="btn btn-crown" type="submit"><i class="fa-solid fa-brain me-2"></i>Analyser l'image</button>
    </form>
    <div id="aiResult" class="mt-3"></div>
</section>

<section class="panel">
    <div class="panel-title">Machine Actions</div>
    <button onclick="sendAction('good_led_on')" class="btn btn-crown me-2 mb-2"><i class="fa-solid fa-lightbulb me-2"></i>Green LED / Good</button>
    <button onclick="sendAction('defect_led_on')" class="btn btn-alert me-2 mb-2"><i class="fa-solid fa-triangle-exclamation me-2"></i>Red LED / Defect</button>
    <button onclick="sendAction('buzzeron')" class="btn btn-alert me-2 mb-2"><i class="fa-solid fa-bell me-2"></i>Buzzer ON</button>
    <button onclick="sendAction('buzzeroff')" class="btn btn-outline-secondary mb-2"><i class="fa-solid fa-bell-slash me-2"></i>Buzzer OFF</button>
    <div class="mt-2 text-muted">Last command: <strong id="lastCommand">None</strong></div>
</section>
</main>
</div>

<script>
const initialLabels = {{ history.timestamps|tojson }};
let cameraRefreshTimer = null;
let autoInspectionTimer = null;
let isPredicting = false;

function setText(id, value){
    const el = document.getElementById(id);
    if(el) el.textContent = value;
}

function sleep(ms){
    return new Promise(resolve => setTimeout(resolve, ms));
}

function refreshPhoneImage(){
    const img = document.getElementById('cameraImage');
    if(img){
        img.src = '/phone-camera-shot?t=' + Date.now();
    }
}

function openCamera(){
    const cameraBadge = document.getElementById('cameraBadge');
    if(cameraBadge){
        cameraBadge.textContent = 'PHONE CAMERA ACTIVE';
    }
    refreshPhoneImage();
    if(cameraRefreshTimer){
        clearInterval(cameraRefreshTimer);
    }
    cameraRefreshTimer = setInterval(refreshPhoneImage, 800);
}

const defectCtx = document.getElementById('defectChart').getContext('2d');
const defectChart = new Chart(defectCtx, {
    type:'doughnut',
    data:{
        labels:['GOOD CANS','DEFECTIVE CANS'],
        datasets:[{
            data:[{{ current.good_cans }}, {{ current.defective_cans }}],
            backgroundColor:['#007a3d', '#d57435'],
            borderWidth:2
        }]
    },
    options:{responsive:true, maintainAspectRatio:false, plugins:{legend:{position:'top'}}}
});

const qualityCtx = document.getElementById('qualityChart').getContext('2d');
const qualityChart = new Chart(qualityCtx, {
    type:'line',
    data:{
        labels:initialLabels,
        datasets:[
            { label:'Accepted %', data:{{ history.good_rates|tojson }}, borderWidth:3, tension:.35 },
            { label:'Defect %', data:{{ history.defect_rates|tojson }}, borderWidth:3, tension:.35 }
        ]
    },
    options:{responsive:true, maintainAspectRatio:false}
});

async function captureAndPredict(){
    const resultBox = document.getElementById('cameraAiResult');
    if(isPredicting) return;
    isPredicting = true;

    if(resultBox){
        resultBox.innerHTML = '<div class="text-muted fw-bold">Analyse depuis caméra téléphone...</div>';
    }

    try{
        const response = await fetch('/predict-image', { method:'POST' });
        const data = await response.json();

        if(data.error){
            if(resultBox) resultBox.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
            return;
        }

        updateDashboardFromPrediction(data);
        showCameraPredictionResult(data);
        refreshDashboard();

    }catch(err){
        console.error(err);
        if(resultBox) resultBox.innerHTML = '<div class="alert alert-danger">Erreur pendant la prédiction caméra téléphone.</div>';
    }finally{
        isPredicting = false;
    }
}

async function capture360AndPredict(){
    const resultBox = document.getElementById('cameraAiResult');
    if(isPredicting) return;
    isPredicting = true;

    if(resultBox){
        resultBox.innerHTML = '<div class="alert alert-info fw-bold">Inspection 360° depuis caméra téléphone. Tourne la canette doucement...</div>';
    }

    try{
        const response = await fetch('/predict-360', { method:'POST' });
        const data = await response.json();

        if(data.error){
            if(resultBox) resultBox.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
            return;
        }

        updateDashboardFromPrediction(data);
        showCameraPredictionResult(data);
        refreshDashboard();

    }catch(err){
        console.error(err);
        if(resultBox) resultBox.innerHTML = '<div class="alert alert-danger">Erreur pendant inspection 360° téléphone.</div>';
    }finally{
        isPredicting = false;
    }
}

function updateDashboardFromPrediction(data){
    setText('currentBrand', data.brand);
    setText('brandConfidence', data.brand_confidence);
    setText('brandBox', data.brand);
    setText('brandConfidenceBox', data.brand_confidence);
    setText('brandClassifier', data.brand);
    setText('defectType', data.defect_type);
    setText('patchcoreResult', data.patchcore_result);
    setText('efficientnetResult', data.efficientnet_result);
    setText('lastInspection', new Date().toLocaleString());

    const qBox = document.getElementById('qualityStatusBox');
    if(qBox){
        qBox.textContent = data.quality_status;
        qBox.className = 'big-status ' + (data.quality_status === 'GOOD' ? 'good' : 'defect');
    }

    const cameraBadge = document.getElementById('cameraBadge');
    if(cameraBadge){
        cameraBadge.textContent = data.brand + ' · ' + data.brand_confidence + '%';
    }
}

function showCameraPredictionResult(data){
    const resultBox = document.getElementById('cameraAiResult');
    if(!resultBox) return;

    const statusClass = data.quality_status === 'GOOD' ? 'good' : 'defect';
    resultBox.innerHTML = `
        <div class="row g-3 align-items-center">
            <div class="col-md-4">
                <img src="${data.image_path}" class="img-fluid rounded-3 border">
            </div>
            <div class="col-md-8">
                <div class="big-status ${statusClass}" style="font-size:1.25rem;padding:12px;">${data.quality_status}</div>
                <div class="info-row"><span>Marque détectée</span><strong>${data.brand} (${data.brand_confidence}%)</strong></div>
                <div class="info-row"><span>Défaut</span><strong>${data.defect_type}</strong></div>
                <div class="info-row"><span>EfficientNet</span><strong>${data.efficientnet_result}</strong></div>
                <div class="info-row"><span>PatchCore</span><strong>${data.patchcore_result}</strong></div>
                ${data.frames_analyzed ? `<div class="info-row"><span>Vues inspectées</span><strong>${data.frames_analyzed} frames / 360°</strong></div>` : ''}
            </div>
        </div>`;
}

function toggleAutoInspection(){
    const btnText = document.getElementById('autoBtnText');

    if(autoInspectionTimer){
        clearInterval(autoInspectionTimer);
        autoInspectionTimer = null;
        if(btnText) btnText.textContent = 'Auto OFF';
        return;
    }

    autoInspectionTimer = setInterval(captureAndPredict, 3000);
    if(btnText) btnText.textContent = 'Auto ON';
    captureAndPredict();
}

function sendAction(action){
    fetch('/action', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({action})
    }).then(r => {
        if(r.ok){
            const last = document.getElementById('lastCommand');
            if(last) last.textContent = action;
            refreshDashboard();
        }
    });
}

const aiUploadForm = document.getElementById('aiUploadForm');
if(aiUploadForm){
    aiUploadForm.addEventListener('submit', async function(e){
        e.preventDefault();
        const fileInput = document.getElementById('aiImage');
        const file = fileInput.files[0];
        if(!file){ alert('Veuillez choisir une image.'); return; }

        const formData = new FormData();
        formData.append('file', file);

        document.getElementById('aiResult').innerHTML = '<div class="text-muted fw-bold">Analyse en cours...</div>';

        try{
            const response = await fetch('/predict-image', {method:'POST', body:formData});
            const data = await response.json();

            if(data.error){
                document.getElementById('aiResult').innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                return;
            }

            const statusClass = data.quality_status === 'GOOD' ? 'good' : 'defect';
            document.getElementById('aiResult').innerHTML = `
                <div class="row g-3 align-items-center">
                    <div class="col-md-4"><img src="${data.image_path}" class="img-fluid rounded-3 border"></div>
                    <div class="col-md-8">
                        <div class="big-status ${statusClass}" style="font-size:1.4rem;padding:14px;">${data.quality_status}</div>
                        <div class="info-row"><span>Marque</span><strong>${data.brand} (${data.brand_confidence}%)</strong></div>
                        <div class="info-row"><span>Défaut</span><strong>${data.defect_type}</strong></div>
                        <div class="info-row"><span>EfficientNet</span><strong>${data.efficientnet_result}</strong></div>
                        <div class="info-row"><span>PatchCore</span><strong>${data.patchcore_result}</strong></div>
                    </div>
                </div>`;
            refreshDashboard();
        }catch(err){
            document.getElementById('aiResult').innerHTML = '<div class="alert alert-danger">Erreur pendant la prédiction.</div>';
            console.error(err);
        }
    });
}

async function refreshDashboard(){
    try{
        const response = await fetch('/quality-data');
        if(!response.ok) return;
        const data = await response.json();
        const c = data.current;

        setText('totalInspected', c.total_inspected);
        setText('goodCans', c.good_cans);
        setText('defectiveCans', c.defective_cans);
        setText('acceptedPercent', c.accepted_percent + '%');
        setText('defectRate', c.defect_rate + '%');
        setText('currentBrand', c.current_brand);
        setText('brandConfidence', c.brand_confidence);
        setText('brandBox', c.current_brand);
        setText('brandConfidenceBox', c.brand_confidence);
        setText('defectType', c.defect_type);
        setText('brandClassifier', c.current_brand);
        setText('patchcoreResult', c.patchcore_result);
        setText('efficientnetResult', c.efficientnet_result);
        setText('lastInspection', c.timestamp);
        setText('aiStatus', c.ai_status);
        setText('cameraStatus', c.camera_status);

        const qBox = document.getElementById('qualityStatusBox');
        if(qBox){
            qBox.textContent = c.quality_status;
            qBox.className = 'big-status ' + (c.quality_status === 'GOOD' ? 'good' : 'defect');
        }

        qualityChart.data.labels = data.history.timestamps;
        qualityChart.data.datasets[0].data = data.history.good_rates;
        qualityChart.data.datasets[1].data = data.history.defect_rates;
        qualityChart.update('none');

        defectChart.data.datasets[0].data = [c.good_cans, c.defective_cans];
        defectChart.update('none');

        const historyTable = document.getElementById('historyTable');
        if(historyTable){
            historyTable.innerHTML = data.history.history_rows.slice().reverse().map(r => `<tr><td>${r.time}</td><td>${r.brand}</td><td>${r.result}</td></tr>`).join('');
        }
    }catch(e){
        console.error('Dashboard refresh error:', e);
    }
}

setInterval(refreshDashboard, 5000);
openCamera();
</script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

CHATBOT_HTML = r"""
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Assistant IA - Crown Dashboard</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
:root{--green:#007a3d;--deep:#174d45;--border:#c8ddd8;--text:#203b36;--panel:#ffffffee;}
body{margin:0;min-height:100vh;background:linear-gradient(135deg,#eef6f2,#fff8df);font-family:"Segoe UI",system-ui,sans-serif;color:var(--text);padding:22px;}
.chat-shell{max-width:980px;margin:auto;background:var(--panel);border:1px solid var(--border);border-radius:22px;box-shadow:0 12px 28px rgba(28,92,83,.14);overflow:hidden;}
.chat-header{padding:18px 22px;background:linear-gradient(135deg,#3f837a,#174d45);color:white;display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;}
.chat-header h1{font-size:1.25rem;font-weight:950;margin:0;}
.messages{height:520px;overflow:auto;padding:18px;background:#f8fbfa;}
.msg{max-width:78%;padding:12px 14px;border-radius:16px;margin-bottom:12px;line-height:1.45;white-space:pre-wrap;}
.user{margin-left:auto;background:#007a3d;color:white;border-bottom-right-radius:4px;}
.bot{margin-right:auto;background:white;border:1px solid var(--border);border-bottom-left-radius:4px;}
.chat-input{padding:14px;background:white;border-top:1px solid var(--border);}
.btn-crown{background:linear-gradient(135deg,var(--green),#0b5f35);border:0;color:white;border-radius:12px;font-weight:800;}
.btn-back{background:white;color:var(--deep);border:1px solid rgba(255,255,255,.6);border-radius:12px;font-weight:800;}
.quick{border:1px solid var(--border);background:white;border-radius:999px;padding:7px 10px;margin:4px;font-size:.86rem;}
</style>
</head>
<body>
<div class="chat-shell">
    <div class="chat-header">
        <div>
            <h1><i class="fa-solid fa-comments me-2"></i>Assistant IA - Deco Machine</h1>
            <small>Chatbot local sans API externe</small>
        </div>
        <a href="/" class="btn btn-back btn-sm"><i class="fa-solid fa-arrow-left me-1"></i>Retour Dashboard</a>
    </div>
    <div id="messages" class="messages">
        <div class="msg bot">Bonjour, je suis l'assistant IA du dashboard Crown. Pose une question sur la marque, les défauts, Missing Paint, Color Contamination ou l'inspection 360°.</div>
    </div>
    <div class="px-3 pb-2 bg-white">
        <button class="quick" onclick="quickAsk('Explique le dernier résultat de détection')">Expliquer dernier résultat</button>
        <button class="quick" onclick="quickAsk('Pourquoi utiliser inspection 360 ?')">Pourquoi 360° ?</button>
        <button class="quick" onclick="quickAsk('Quelle différence entre Missing Paint et Color Contamination ?')">Différence défauts</button>
    </div>
    <div class="chat-input">
        <div class="input-group">
            <input id="userInput" class="form-control" placeholder="Écris ta question..." onkeydown="if(event.key==='Enter'){sendMessage();}">
            <button class="btn btn-crown" onclick="sendMessage()"><i class="fa-solid fa-paper-plane me-1"></i>Envoyer</button>
        </div>
    </div>
</div>
<script>
function escapeHtml(text){ const div=document.createElement('div'); div.textContent=text; return div.innerHTML; }
function addMessage(text, cls){ const box=document.getElementById('messages'); const div=document.createElement('div'); div.className='msg '+cls; div.innerHTML=escapeHtml(text); box.appendChild(div); box.scrollTop=box.scrollHeight; return div; }
function quickAsk(text){ document.getElementById('userInput').value=text; sendMessage(); }
async function sendMessage(){
    const input=document.getElementById('userInput');
    const msg=input.value.trim();
    if(!msg) return;
    addMessage(msg,'user');
    input.value='';
    const loading=addMessage('Réponse en cours...','bot');
    try{
        const response=await fetch('/chatbot-api',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:msg})});
        const data=await response.json();
        loading.innerHTML=escapeHtml(data.reply || 'Aucune réponse.');
    }catch(e){
        loading.innerHTML='Erreur de connexion avec le chatbot.';
    }
}
</script>
</body>
</html>
"""

CONTROL_HTML = r"""
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Control Center - Crown</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
:root{--green:#007a3d;--deep:#174d45;--orange:#d57435;--border:#c8ddd8;--panel:#ffffffee;}
body{background:linear-gradient(135deg,#eef6f2,#fff8df);font-family:"Segoe UI",system-ui,sans-serif;padding:20px;color:#203b36;}
.header,.metric-card,.command-box{background:var(--panel);border:1px solid var(--border);border-radius:18px;box-shadow:0 12px 28px rgba(28,92,83,.12);padding:18px;margin-bottom:16px;}
.metric-label{color:#6b7d78;font-weight:800;font-size:.8rem;}
.metric-value{font-size:1.8rem;font-weight:950;color:var(--deep);}
.metric-sub{font-size:.85rem;color:#6b7d78;}
.btn-control{display:block;width:100%;border:0;border-radius:12px;font-weight:900;padding:12px;margin-bottom:10px;color:white;}
.btn-start{background:#007a3d}.btn-stop{background:#c0392b}.btn-pause{background:#d57435}.btn-blue{background:#2f7468}.btn-dark-custom{background:#174d45}.btn-orange{background:#d57435}
.info-line{display:flex;justify-content:space-between;border-bottom:1px solid #e2eeeb;padding:9px 0;}
.badge-online{background:#e7f7ef;color:#08783f;border-radius:999px;padding:5px 9px;}
.log-box{background:#0f2b27;color:#dff9ef;border-radius:14px;padding:14px;min-height:140px;font-family:Consolas,monospace;}
.status-pill{background:#f4fbf7;border:1px solid var(--border);border-radius:999px;padding:7px 10px;font-weight:700;font-size:.78rem;}
.dot{width:9px;height:9px;display:inline-block;border-radius:50%;background:#17a75a;margin-right:6px;}
</style>
</head>
<body>
<div class="header d-flex justify-content-between align-items-center flex-wrap gap-3">
    <div>
        <h2 class="fw-bold m-0">CROWN MAGHREB - Machine Control</h2>
        <div class="text-muted">Deco Machine Quality Inspection Control Center</div>
    </div>
    <div class="d-flex gap-2 flex-wrap align-items-center justify-content-end">
        <span class="status-pill"><span class="dot"></span>Phone Camera</span>
        <span class="status-pill"><span class="dot"></span>AI Online</span>
        <span class="status-pill"><span class="dot"></span>Raspberry Pi Connected</span>
        <a href="/" class="btn btn-dark btn-sm">Dashboard</a>
    </div>
</div>

<div class="row g-3">
    <div class="col-md-3"><div class="metric-card"><div class="metric-label">Quality Status</div><div class="metric-value" id="qualityValue">{{ current.quality_status }}</div><div class="metric-sub">Current defect: <strong id="defectTypeValue">{{ current.defect_type }}</strong></div></div></div>
    <div class="col-md-3"><div class="metric-card"><div class="metric-label">Inspection Mode</div><div class="metric-value" id="modeValue">AUTO</div><div class="metric-sub">Deco Machine Line</div></div></div>
    <div class="col-md-3"><div class="metric-card"><div class="metric-label">Batch ID</div><div class="metric-value" style="font-size:1.25rem;" id="batchValue">{{ batch_id }}</div><div class="metric-sub">Dynamic production batch</div></div></div>
    <div class="col-md-3"><div class="metric-card"><div class="metric-label">Last Inspection</div><div class="metric-value" style="font-size:1.1rem;" id="lastInspection">{{ current.timestamp }}</div><div class="metric-sub">Live dashboard synchronization</div></div></div>
    <div class="col-md-3"><div class="metric-card"><div class="metric-label">Accepted</div><div class="metric-value" id="goodCans">{{ current.good_cans }}</div><div class="metric-sub"><span id="acceptedPercent">{{ current.accepted_percent }}</span>% accepted</div></div></div>
    <div class="col-md-3"><div class="metric-card"><div class="metric-label">Rejected</div><div class="metric-value" style="color:var(--orange);" id="defectiveCans">{{ current.defective_cans }}</div><div class="metric-sub"><span id="defectRate">{{ current.defect_rate }}</span>% defect rate</div></div></div>

    <div class="col-lg-4">
        <div class="command-box">
            <h5 class="fw-bold">Inspection Commands</h5>
            <button onclick="sendAction('start_inspection')" class="btn-control btn-start"><i class="fa-solid fa-play me-2"></i>Start Inspection</button>
            <button onclick="sendAction('pause_inspection')" class="btn-control btn-pause"><i class="fa-solid fa-pause me-2"></i>Pause Inspection</button>
            <button onclick="sendAction('stop_inspection')" class="btn-control btn-stop"><i class="fa-solid fa-stop me-2"></i>Stop Inspection</button>
            <button onclick="sendAction('reset_counters')" class="btn-control btn-dark-custom"><i class="fa-solid fa-rotate-left me-2"></i>Reset Counters</button>
        </div>
    </div>

    <div class="col-lg-4">
        <div class="command-box">
            <h5 class="fw-bold">AI / System</h5>
            <div class="info-line"><span>Brand Classifier</span><strong class="badge-online">ONLINE</strong></div>
            <div class="info-line"><span>PatchCore</span><strong id="patchcoreBadge" class="badge-online">{{ current.patchcore_result }}</strong></div>
            <div class="info-line"><span>EfficientNetB0</span><strong id="efficientnetBadge" class="badge-online">{{ current.efficientnet_result }}</strong></div>
            <button onclick="sendAction('reload_models')" class="btn-control btn-blue mt-3"><i class="fa-solid fa-brain me-2"></i>Reload AI Models</button>
            <button onclick="sendAction('test_camera')" class="btn-control btn-blue"><i class="fa-solid fa-camera me-2"></i>Test Camera</button>
        </div>
    </div>

    <div class="col-lg-4">
        <div class="command-box">
            <h5 class="fw-bold">Simulation / Signals</h5>
            <button onclick="sendAction('simulate_good')" class="btn-control btn-start"><i class="fa-solid fa-circle-check me-2"></i>Simulate GOOD Can</button>
            <button onclick="sendAction('simulate_missing_paint')" class="btn-control btn-stop"><i class="fa-solid fa-paint-roller me-2"></i>Simulate Missing Paint</button>
            <button onclick="sendAction('simulate_contamination')" class="btn-control btn-orange"><i class="fa-solid fa-droplet me-2"></i>Simulate Color Contamination</button>
            <button onclick="sendAction('green_led')" class="btn-control btn-start"><i class="fa-solid fa-lightbulb me-2"></i>Green LED / GOOD Signal</button>
            <button onclick="sendAction('red_led')" class="btn-control btn-stop"><i class="fa-solid fa-triangle-exclamation me-2"></i>Red LED / DEFECT Signal</button>
            <button onclick="sendAction('buzzer')" class="btn-control btn-orange"><i class="fa-solid fa-bell me-2"></i>Buzzer Alert</button>
            <button onclick="sendAction('reject_can')" class="btn-control btn-dark-custom"><i class="fa-solid fa-ban me-2"></i>Reject Can Signal</button>
        </div>
    </div>

    <div class="col-lg-12">
        <div class="command-box">
            <div class="text-muted fw-bold mb-2">SYSTEM LOG</div>
            <div id="systemLog" class="log-box"><div class="log-row">System initialized and synchronized with dashboard</div></div>
        </div>
    </div>
</div>

<script>
function setText(id, value){ const el=document.getElementById(id); if(el) el.textContent=value; }
function addLog(text){
    const log=document.getElementById('systemLog');
    const div=document.createElement('div');
    div.textContent='[' + new Date().toLocaleTimeString() + '] ' + text;
    log.prepend(div);
}
async function sendAction(action){
    const response = await fetch('/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action})});
    const data = await response.json();
    addLog('Action sent: ' + action);
    refreshControl();
}
async function refreshControl(){
    const response = await fetch('/quality-data');
    if(!response.ok) return;
    const data = await response.json();
    const c = data.current;
    setText('qualityValue', c.quality_status);
    setText('defectTypeValue', c.defect_type);
    setText('goodCans', c.good_cans);
    setText('defectiveCans', c.defective_cans);
    setText('acceptedPercent', c.accepted_percent);
    setText('defectRate', c.defect_rate);
    setText('lastInspection', c.timestamp);
    setText('patchcoreBadge', c.patchcore_result);
    setText('efficientnetBadge', c.efficientnet_result);
}
setInterval(refreshControl, 5000);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
