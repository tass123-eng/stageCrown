from flask import Flask, render_template_string, request, jsonify, send_file
from datetime import datetime
from pathlib import Path
import random
import threading
import time
import socket
import os
import numpy as np
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

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
# Raspberry / socket config
server_ip = "192.168.48.238"
server_port_socket = 6000

LOGO_FILE = BASE_DIR / "crown_logo.png"
IMAGE_URL = os.getenv(
    "DASHBOARD_IMAGE_URL",
    "https://images.unsplash.com/photo-1581092335397-9583eb92d232?auto=format&fit=crop&w=1200&q=80"
)
# =========================
# AI models config
# =========================
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

BRAND_MODEL_PATH = BASE_DIR / "models" / "crown_clean_final.keras"
DEFECT_MODEL_PATH = BASE_DIR / "models" / "final_defect_classifier.keras"

BRAND_CLASSES = [
    "amestel beer", "apla", "arwa ananas", "arwa mandarine",
    "boga lemon", "celtia", "coca cola", "fanta",
    "hamoud", "selcto", "slim", "sprite", "viva citron"
]

# IMPORTANT : ordre réel du modèle final_defect_classifier.keras
# Correction : good et color_contamination étaient inversés.
# IMPORTANT : ordre réel le plus probable avec image_dataset_from_directory / flow_from_directory
# Les dossiers sont classés alphabétiquement : good, pb color_contamination, pb missing_paint.
# NE PAS inverser good et color_contamination sans vérifier class_indices pendant l'entraînement.
DEFECT_CLASSES = [
    "good",
    "pb color_contamination",
    "pb missing_paint"
]

# Seuils de sécurité pour éviter de valider GOOD trop vite.
# En production: GOOD final uniquement après inspection multi-vues / 360°.
DEFECT_MIN_CONFIDENCE = 55.0
GOOD_MIN_CONFIDENCE = 75.0
MIN_MARGIN_CONFIDENCE = 8.0
MISSING_PAINT_CLOSE_MARGIN = 10.0
INSPECTION_360_FRAMES = 8

brand_model = None
defect_model = None

def load_ai_models():
    global brand_model, defect_model
    if not TENSORFLOW_AVAILABLE:
        current_state["ai_status"] = "TensorFlow Missing" if "current_state" in globals() else "TensorFlow Missing"
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

    except Exception as e:
        print("❌ AI models loading error:", e)


def prepare_image_for_model(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_brand_ai(img_path):
    if brand_model is None:
        raise RuntimeError("Brand model not loaded. Check models/crown_clean_final.keras")
    img_array = prepare_image_for_model(img_path, (224, 224))
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
    """
    Prédiction défaut corrigée.
    Objectif : éviter que toutes les canettes défectueuses soient forcées en color_contamination.

    Règles :
    1) GOOD seulement si le score good est fort ET clairement supérieur aux deux défauts.
    2) DEFECT si un défaut est suffisamment fort.
    3) Si color_contamination et missing_paint sont proches, on favorise missing_paint
       au lieu de classer automatiquement color_contamination.
    4) Si le modèle hésite, on affiche SUSPECT / À vérifier 360°.
    """
    if defect_model is None:
        raise RuntimeError("Defect model not loaded. Check models/final_defect_classifier.keras")

    img_array = prepare_image_for_model(img_path, (300, 300))
    preds = defect_model.predict(img_array, verbose=0)[0]

    if len(preds) != len(DEFECT_CLASSES):
        raise RuntimeError(
            f"Nombre de sorties modèle ({len(preds)}) différent de DEFECT_CLASSES ({len(DEFECT_CLASSES)}). "
            "Vérifie l'ordre des classes utilisé pendant l'entraînement."
        )

    scores = {cls: float(preds[i] * 100.0) for i, cls in enumerate(DEFECT_CLASSES)}

    good_score = scores.get("good", 0.0)
    color_score = scores.get("pb color_contamination", 0.0)
    missing_score = scores.get("pb missing_paint", 0.0)

    best_defect_score = max(color_score, missing_score)

    print("\n========== DEFECT PREDICTION ==========")
    print("Ordre DEFECT_CLASSES utilisé:", DEFECT_CLASSES)
    print("Raw probabilities:")
    for cls in DEFECT_CLASSES:
        print(f"{cls:25s}: {scores[cls]:.2f}%")

    # 1) Validation GOOD très stricte
    if good_score >= GOOD_MIN_CONFIDENCE and (good_score - best_defect_score) >= MIN_MARGIN_CONFIDENCE:
        raw_label = "good"
        confidence = round(good_score, 2)
        print("Decision corrected: GOOD")
        print("=======================================\n")
        defect_type, efficientnet_result, quality_status, patchcore_result = normalize_defect_label(raw_label)
        return raw_label, defect_type, efficientnet_result, quality_status, patchcore_result, confidence, scores

    # 2) Si les deux défauts sont faibles, ne pas forcer une classe
    if best_defect_score < DEFECT_MIN_CONFIDENCE:
        raw_label = "suspect"
        confidence = round(max(good_score, best_defect_score), 2)
        print("Decision corrected: SUSPECT, defect scores too low")
        print("=======================================\n")
        return raw_label, "À vérifier / 360°", "Suspect", "SUSPECT", "Need 360°", confidence, scores

    # 3) Choix du type défaut : missing_paint vs color_contamination
    # Si missing_paint est proche de contamination, on évite de tout classer en contamination.
    if missing_score >= color_score:
        raw_label = "pb missing_paint"
        confidence = round(missing_score, 2)
    elif (color_score - missing_score) <= MISSING_PAINT_CLOSE_MARGIN and missing_score >= DEFECT_MIN_CONFIDENCE:
        raw_label = "pb missing_paint"
        confidence = round(missing_score, 2)
    else:
        raw_label = "pb color_contamination"
        confidence = round(color_score, 2)

    # 4) Si la marge entre les deux défauts est trop petite, afficher suspect sauf si missing a été choisi par règle proche.
    margin = abs(color_score - missing_score)
    if margin < 5.0:
        raw_label = "suspect"
        confidence = round(best_defect_score, 2)
        print("Decision corrected: SUSPECT, color/missing too close")
        print("=======================================\n")
        return raw_label, "Défaut à confirmer / 360°", "Suspect", "SUSPECT", "Need 360°", confidence, scores

    print("Decision corrected:", raw_label)
    print("Confidence:", confidence, "%")
    print("=======================================\n")

    defect_type, efficientnet_result, quality_status, patchcore_result = normalize_defect_label(raw_label)
    return raw_label, defect_type, efficientnet_result, quality_status, patchcore_result, confidence, scores


def update_state_from_prediction(result, count_as_one=True):
    """Met à jour le dashboard sans dupliquer la logique entre image simple et 360°."""
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
            # SUSPECT est compté comme rejet/suspicion pour ne pas laisser passer une canette douteuse.
            current_state["defective_cans"] += 1

    recalculate_percentages()
    add_history_row(
        current_state["current_brand"],
        "Good" if result["quality_status"] == "GOOD" else result["defect_type"]
    )


def predict_full_image(img_path):
    """Prédiction sur une seule image. Utile pour test manuel, mais pas suffisante pour valider GOOD."""
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
    """
    Décision finale 360° corrigée.

    Ancienne erreur:
    - Si une frame était SUSPECT, la décision finale devenait SUSPECT même avec 83% de défaut.
    - Donc une canette réellement défectueuse pouvait rester "À vérifier".

    Nouvelle logique:
    1) On agrège les scores des 8 vues.
    2) Si au moins une vue confirme un défaut fiable -> DEFECT.
    3) Sinon, si les scores 360° montrent un défaut fort -> DEFECT.
    4) GOOD seulement si toutes les vues sont GOOD avec un score good fort.
    5) Sinon -> SUSPECT.
    """
    if not results:
        raise RuntimeError("Aucune frame 360° reçue")

    best_brand = max(results, key=lambda r: r["brand_confidence"])

    # Frames avec défaut déjà confirmé par predict_defect_ai()
    defect_frames = [
        r for r in results
        if r.get("quality_status") == "DEFECT" and float(r.get("defect_confidence", 0)) >= DEFECT_MIN_CONFIDENCE
    ]

    if defect_frames:
        best_defect = max(defect_frames, key=lambda r: float(r.get("defect_confidence", 0)))
        return {
            "brand": best_brand["brand"],
            "brand_confidence": best_brand["brand_confidence"],
            "raw_defect": best_defect["raw_defect"],
            "defect_type": best_defect["defect_type"],
            "defect_confidence": best_defect["defect_confidence"],
            "defect_scores": best_defect.get("defect_scores", {}),
            "quality_status": "DEFECT",
            "patchcore_result": "Anomaly",
            "efficientnet_result": best_defect["efficientnet_result"],
            "image_path": best_defect["image_path"],
            "frames_analyzed": len(results),
            "decision_rule": "DEFECT confirmé: au moins une vue 360° est defectueuse",
            "frames": results,
        }

    # Agrégation des scores sur toutes les vues, même si elles sont SUSPECT.
    good_scores = []
    color_scores = []
    missing_scores = []

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

    # Meilleur défaut observé pendant le tour 360°
    if max_color >= max_missing:
        best_defect_label = "pb color_contamination"
        best_defect_type = "Color Contamination"
        best_defect_conf = round(max_color, 2)
        avg_best_defect = avg_color
    else:
        best_defect_label = "pb missing_paint"
        best_defect_type = "Missing Paint"
        best_defect_conf = round(max_missing, 2)
        avg_best_defect = avg_missing

    # Frame image la plus représentative du défaut choisi
    if best_defect_label == "pb color_contamination":
        best_frame = max(results, key=lambda r: float((r.get("defect_scores", {}) or {}).get("pb color_contamination", 0.0)))
    else:
        best_frame = max(results, key=lambda r: float((r.get("defect_scores", {}) or {}).get("pb missing_paint", 0.0)))

    # IMPORTANT:
    # Si la meilleure vue a un défaut >= 80%, on confirme DEFECT.
    # Cela corrige ton cas: Sprite 83.48% ne doit pas rester SUSPECT.
    if best_defect_conf >= 80.0:
        return {
            "brand": best_brand["brand"],
            "brand_confidence": best_brand["brand_confidence"],
            "raw_defect": best_defect_label,
            "defect_type": best_defect_type,
            "defect_confidence": best_defect_conf,
            "defect_scores": {
                "avg_good": round(avg_good, 2),
                "avg_color_contamination": round(avg_color, 2),
                "avg_missing_paint": round(avg_missing, 2),
                "max_good": round(max_good, 2),
                "max_color_contamination": round(max_color, 2),
                "max_missing_paint": round(max_missing, 2),
            },
            "quality_status": "DEFECT",
            "patchcore_result": "Anomaly",
            "efficientnet_result": best_defect_type,
            "image_path": best_frame["image_path"],
            "frames_analyzed": len(results),
            "decision_rule": "DEFECT confirmé après 360°: une vue présente un défaut avec confiance >= 80%",
            "frames": results,
        }

    # Si le défaut est répétitif sur plusieurs vues, on confirme même si chaque vue est moyenne.
    if avg_best_defect >= 60.0 and avg_best_defect > avg_good:
        return {
            "brand": best_brand["brand"],
            "brand_confidence": best_brand["brand_confidence"],
            "raw_defect": best_defect_label,
            "defect_type": best_defect_type,
            "defect_confidence": round(avg_best_defect, 2),
            "defect_scores": {
                "avg_good": round(avg_good, 2),
                "avg_color_contamination": round(avg_color, 2),
                "avg_missing_paint": round(avg_missing, 2),
                "max_good": round(max_good, 2),
                "max_color_contamination": round(max_color, 2),
                "max_missing_paint": round(max_missing, 2),
            },
            "quality_status": "DEFECT",
            "patchcore_result": "Anomaly",
            "efficientnet_result": best_defect_type,
            "image_path": best_frame["image_path"],
            "frames_analyzed": len(results),
            "decision_rule": "DEFECT confirmé: défaut répété sur plusieurs vues 360°",
            "frames": results,
        }

    # ------------------------------------------------------------------
    # OVERRIDE IMPORTANT POUR TON CAS RÉEL
    # ------------------------------------------------------------------
    # Ton modèle donne parfois: Good ~86%, Missing Paint ~8%, Contamination ~5%.
    # Donc il met la canette en GOOD/SUSPECT alors que visuellement elle est Missing Paint.
    # Cette règle empêche de valider GOOD quand le score good n'est pas très fort
    # et qu'un défaut ressort comme 2e signal pendant le 360°.
    # Elle aide aussi les contaminations classées à tort comme GOOD.
    defect_signal_conf = max(max_color, max_missing)
    defect_signal_avg = max(avg_color, avg_missing)
    good_is_not_solid = avg_good < 92.0 or max_good < 92.0
    defect_is_visible_signal = defect_signal_conf >= 7.0 or defect_signal_avg >= 5.0

    if good_is_not_solid and defect_is_visible_signal:
        # on choisit le type du défaut qui a le plus grand signal
        if max_missing >= max_color:
            override_label = "pb missing_paint"
            override_type = "Missing Paint"
            override_conf = round(max_missing, 2)
            override_frame = max(results, key=lambda r: float((r.get("defect_scores", {}) or {}).get("pb missing_paint", 0.0)))
        else:
            override_label = "pb color_contamination"
            override_type = "Color Contamination"
            override_conf = round(max_color, 2)
            override_frame = max(results, key=lambda r: float((r.get("defect_scores", {}) or {}).get("pb color_contamination", 0.0)))

        return {
            "brand": best_brand["brand"],
            "brand_confidence": best_brand["brand_confidence"],
            "raw_defect": override_label,
            "defect_type": override_type,
            "defect_confidence": override_conf,
            "defect_scores": {
                "avg_good": round(avg_good, 2),
                "avg_color_contamination": round(avg_color, 2),
                "avg_missing_paint": round(avg_missing, 2),
                "max_good": round(max_good, 2),
                "max_color_contamination": round(max_color, 2),
                "max_missing_paint": round(max_missing, 2),
            },
            "quality_status": "DEFECT",
            "patchcore_result": "Anomaly",
            "efficientnet_result": override_type,
            "image_path": override_frame["image_path"],
            "frames_analyzed": len(results),
            "decision_rule": "DEFECT confirmé par override 360°: le score GOOD n'est pas solide et un signal défaut est visible",
            "frames": results,
        }

    # GOOD uniquement si toutes les frames sont GOOD et que le score good domine.
    all_good = all(r.get("quality_status") == "GOOD" for r in results)
    if all_good and avg_good >= GOOD_MIN_CONFIDENCE and avg_good > avg_best_defect:
        best_good = max(results, key=lambda r: float((r.get("defect_scores", {}) or {}).get("good", r.get("defect_confidence", 0))))
        return {
            "brand": best_brand["brand"],
            "brand_confidence": best_brand["brand_confidence"],
            "raw_defect": "good",
            "defect_type": "None",
            "defect_confidence": round(avg_good, 2),
            "defect_scores": {
                "avg_good": round(avg_good, 2),
                "avg_color_contamination": round(avg_color, 2),
                "avg_missing_paint": round(avg_missing, 2),
                "max_good": round(max_good, 2),
                "max_color_contamination": round(max_color, 2),
                "max_missing_paint": round(max_missing, 2),
            },
            "quality_status": "GOOD",
            "patchcore_result": "Normal",
            "efficientnet_result": "Good",
            "image_path": best_good["image_path"],
            "frames_analyzed": len(results),
            "decision_rule": "GOOD confirmé: toutes les vues 360° sont GOOD",
            "frames": results,
        }

    return {
        "brand": best_brand["brand"],
        "brand_confidence": best_brand["brand_confidence"],
        "raw_defect": "suspect",
        "defect_type": "À vérifier / refaire 360° plus proche",
        "defect_confidence": round(max(best_defect_conf, max_good), 2),
        "defect_scores": {
            "avg_good": round(avg_good, 2),
            "avg_color_contamination": round(avg_color, 2),
            "avg_missing_paint": round(avg_missing, 2),
            "max_good": round(max_good, 2),
            "max_color_contamination": round(max_color, 2),
            "max_missing_paint": round(max_missing, 2),
        },
        "quality_status": "SUSPECT",
        "patchcore_result": "Need verification",
        "efficientnet_result": "Suspect",
        "image_path": best_frame["image_path"],
        "frames_analyzed": len(results),
        "decision_rule": "SUSPECT: le modèle voit encore trop de fond ou les scores défaut sont faibles. Place la canette dans le rectangle vert et refais 360°.",
        "frames": results,
    }

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
    "camera_status": "Connected",
    "ai_status": "Models Online",
    "raspberry_status": "Connected",
    "database_status": "Connected / Local mode",
}

load_ai_models()

historical_data = {
    "timestamps": [],
    "good_rates": [],
    "defect_rates": [],
    "history_rows": [],
}

def recalculate_percentages():
    total = max(1, current_state["total_inspected"])
    current_state["accepted_percent"] = round((current_state["good_cans"] / total) * 100, 2)
    current_state["defect_rate"] = round((current_state["defective_cans"] / total) * 100, 2)


def update_quality_data():
    """Simulation temps réel. Remplacer cette fonction par les sorties réelles des modèles IA."""
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
            defect_confidence = round(random.uniform(91.0, 98.8), 1)
        else:
            defect_type = "None"
            current_state["good_cans"] = current_state["total_inspected"] - current_state["defective_cans"]
            quality_status = "GOOD"
            patchcore_result = "Normal"
            efficientnet_result = "Good"
            defect_confidence = round(random.uniform(96.0, 99.5), 1)

        current_state["good_cans"] = current_state["total_inspected"] - current_state["defective_cans"]
        current_state.update({
            "timestamp": current_timestamp,
            "image_url": IMAGE_URL,
            "current_brand": brand,
            "brand_confidence": round(random.uniform(97.0, 99.9), 1),
            "quality_status": quality_status,
            "defect_type": defect_type,
            "defect_confidence": defect_confidence,
            "patchcore_result": patchcore_result,
            "efficientnet_result": efficientnet_result,
            "line_status": "Production Running",
            "camera_status": "Connected",
            "ai_status": "Models Online",
            "raspberry_status": "Connected",
        })
        recalculate_percentages()

        historical_data["timestamps"].append(current_timestamp)
        historical_data["good_rates"].append(current_state["accepted_percent"])
        historical_data["defect_rates"].append(current_state["defect_rate"])
        historical_data["history_rows"].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "brand": brand,
            "result": defect_type if is_defect else "Good",
        })

        max_points = 50
        for key in historical_data:
            historical_data[key] = historical_data[key][-max_points:]

        time.sleep(5)


def send_to_server(command):
    """Envoyer une commande au serveur socket Raspberry Pi / machine."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as clientsocket:
            clientsocket.settimeout(3)
            clientsocket.connect((server_ip, server_port_socket))
            clientsocket.sendall(command.encode())
    except Exception as e:
        print("Erreur d'envoi socket :", e)

# Mettre True seulement si tu veux garder la simulation random.
# Pour utiliser les vrais modèles avec la caméra, il faut False.
SIMULATION_MODE = False

if SIMULATION_MODE:
    threading.Thread(target=update_quality_data, daemon=True).start()

@app.route("/logo")
def logo():
    if LOGO_FILE.exists():
        return send_file(LOGO_FILE)
    return "", 404

@app.route("/")
def dashboard():
    defect_distribution = {"Color Contamination": 62, "Missing Paint": 38}
    brand_stats = [
        {"brand": "Coca-Cola", "produced": 2500, "defects": 35},
        {"brand": "Fanta", "produced": 1800, "defects": 42},
        {"brand": "Hamoud", "produced": 1200, "defects": 18},
        {"brand": "Sprite", "produced": 900, "defects": 11},
    ]

    return render_template_string("""
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
        .panel-title { font-size:.98rem; font-weight:900; color:var(--deep); margin-bottom:12px; display:flex; align-items:center; justify-content:space-between; }
        .camera-frame { height:285px; border-radius:16px; overflow:hidden; border:1px solid var(--border); background:#ddebe7; position:relative; }
        .camera-frame img,.camera-frame video { width:100%; height:100%; object-fit:cover; }
        #cameraVideo { display:none; }
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
        <a href="/control" class="nav-line" style="display:block;color:white;text-decoration:none;">
    <i class="fa-solid fa-screwdriver-wrench me-2"></i>Machine Control
</a>
        <div class="mt-4 p-3 rounded-3" style="background:rgba(255,255,255,.12)"><div style="font-size:.75rem;opacity:.75">Project</div><strong>Deco Machine Quality Inspection</strong><div style="font-size:.78rem;opacity:.8;margin-top:6px">Brand recognition + defect detection</div></div>
    </aside>

    <main>
        <section class="header">
            <div>
                <h1>CROWN MAGHREB<br>DECO MACHINE QUALITY INSPECTION SYSTEM</h1>
                <div class="subtitle">AI-Based Brand Recognition & Defect Detection</div>
            </div>
            <div class="status">
                <div class="status-pill"><span class="dot"></span>Camera Connected</div>
                <div class="status-pill"><span class="dot"></span>AI Models Online</div>
                <div class="status-pill"><span class="dot"></span>Raspberry Pi Connected</div>
                <div class="status-pill"><span class="dot"></span>Production Running</div>
                <a href="/control" class="btn btn-crown btn-sm">
                    <i class="fa-solid fa-sliders me-1"></i>Control Center
                </a>
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
                        <button class="btn btn-crown btn-sm" onclick="openCamera()">
                            <i class="fa-solid fa-video me-1"></i>Open Camera
                        </button>
                        <button class="btn btn-alert btn-sm" onclick="captureAndPredict()">
                            <i class="fa-solid fa-brain me-1"></i>Détecter marque
                        </button>
                        <button class="btn btn-crown btn-sm" onclick="capture360AndPredict()">
                            <i class="fa-solid fa-arrows-rotate me-1"></i>Vérifier défectueuse ou pas
                        </button>
                        <button class="btn btn-outline-secondary btn-sm" onclick="toggleAutoInspection()">
                            <i class="fa-solid fa-rotate me-1"></i><span id="autoBtnText">Auto OFF</span>
                        </button>
                    </div>
                </div>
                <div class="camera-frame">
                    <img src="{{ current.image_url }}" id="cameraImage" alt="Live camera feed">
                    <video id="cameraVideo" autoplay playsinline muted></video>
                    <div class="roi-guide"></div>
                    <div class="roi-text">PLACE CAN HERE</div>
                    <div class="badge-live" id="cameraBadge">LIVE CAMERA FEED</div>
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
                <table class="table align-middle"><thead><tr><th>Module</th><th>Résultat</th></tr></thead><tbody>
                    <tr><td>Brand Classifier</td><td id="brandClassifier">{{ current.current_brand }}</td></tr>
                    <tr><td>PatchCore</td><td id="patchcoreResult">{{ current.patchcore_result }}</td></tr>
                    <tr><td>EfficientNetB0</td><td id="efficientnetResult">{{ current.efficientnet_result }}</td></tr>
                </tbody></table>
                <div class="panel-title mt-3"><i class="fa-solid fa-microchip me-2"></i>System Information</div>
                <div class="info-row"><span>Hardware</span><strong>Raspberry Pi 4</strong></div>
                <div class="info-row"><span>Camera Status</span><strong id="cameraStatus">{{ current.camera_status }}</strong></div>
                <div class="info-row"><span>AI Status</span><strong id="aiStatus">Running</strong></div>
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
                <table class="table table-hover align-middle mb-0"><thead><tr><th>Brand</th><th>Produced</th><th>Defects</th><th>Defect %</th></tr></thead><tbody>
                {% for row in brand_stats %}<tr><td>{{ row.brand }}</td><td>{{ row.produced }}</td><td>{{ row.defects }}</td><td>{{ ((row.defects / row.produced) * 100) | round(2) }}%</td></tr>{% endfor %}
                </tbody></table>
            </div>
            <div class="panel">
                <div class="panel-title">Historique des Dernières Détections</div>
                <table class="table table-hover align-middle mb-0"><thead><tr><th>Time</th><th>Brand</th><th>Result</th></tr></thead><tbody id="historyTable">
                {% for row in history_rows|reverse %}<tr><td>{{ row.time }}</td><td>{{ row.brand }}</td><td>{{ row.result }}</td></tr>{% endfor %}
                </tbody></table>
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
const defectCtx = document.getElementById('defectChart').getContext('2d');
const defectChart = new Chart(defectCtx, {
    type:'doughnut',
    data:{
        labels:['GOOD CANS','DEFECTIVE CANS'],
        datasets:[{
            data:[
                {{ current.good_cans }},
                {{ current.defective_cans }}
            ],
            backgroundColor:[
                '#007a3d',
                '#d57435'
            ],
            borderWidth:2
        }]
    },
    options:{
        responsive:true,
        maintainAspectRatio:false,
        plugins:{
            legend:{
                position:'top'
            }
        }
    }
});
const qualityCtx = document.getElementById('qualityChart').getContext('2d');
const qualityChart = new Chart(qualityCtx, { type:'line', data:{ labels:initialLabels, datasets:[{ label:'Accepted %', data:{{ history.good_rates|tojson }}, borderWidth:3, tension:.35 }, { label:'Defect %', data:{{ history.defect_rates|tojson }}, borderWidth:3, tension:.35 }] }, options:{ responsive:true, maintainAspectRatio:false } });

let cameraStream = null;
let autoInspectionTimer = null;
let isPredicting = false;

function sleep(ms){ return new Promise(resolve => setTimeout(resolve, ms)); }

function captureVideoBlob(video, quality=0.92){
    return new Promise((resolve) => {
        // IMPORTANT: on envoie au modèle uniquement la zone de la canette, pas toute l'image.
        // La zone correspond au rectangle vert affiché sur la caméra.
        const vw = video.videoWidth;
        const vh = video.videoHeight;

        const sx = Math.round(vw * 0.35);
        const sy = Math.round(vh * 0.07);
        const sw = Math.round(vw * 0.30);
        const sh = Math.round(vh * 0.86);

        const canvas = document.createElement('canvas');
        canvas.width = 360;
        canvas.height = 720;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(blob => resolve(blob), 'image/jpeg', quality);
    });
}

async function openCamera(){
    const cameraImage = document.getElementById('cameraImage');
    const cameraVideo = document.getElementById('cameraVideo');
    const cameraBadge = document.getElementById('cameraBadge');

    if(!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia){
        alert('Camera not supported.');
        return;
    }

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: "environment"
            },
            audio: false
        });

        cameraVideo.srcObject = cameraStream;
        cameraImage.style.display = 'none';
        cameraVideo.style.display = 'block';

        if(cameraBadge){
            cameraBadge.textContent = 'CAMERA ACTIVE';
        }

    } catch(e){
        alert("Impossible d'ouvrir la caméra. Vérifiez les permissions.");
        console.error(e);
    }
}

async function captureAndPredict(){
    const video = document.getElementById('cameraVideo');
    const resultBox = document.getElementById('cameraAiResult');

    if(!video || !video.srcObject){
        alert("Clique d'abord sur Open Camera.");
        return;
    }

    if(video.videoWidth === 0 || video.videoHeight === 0){
        alert("La caméra n'est pas encore prête. Attends 1 seconde puis réessaie.");
        return;
    }

    if(isPredicting){
        return;
    }

    isPredicting = true;

    if(resultBox){
        resultBox.innerHTML = '<div class="text-muted fw-bold">Analyse de la frame en cours...</div>';
    }

    try{
        const blob = await captureVideoBlob(video, 0.92);
        const formData = new FormData();
        formData.append('file', blob, 'camera_roi.jpg');

        const response = await fetch('/predict-image', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if(data.error){
            if(resultBox){
                resultBox.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
            }
            return;
        }

        updateDashboardFromPrediction(data);
        showCameraPredictionResult(data);
        refreshDashboard();

    } catch(err){
        console.error(err);
        if(resultBox){
            resultBox.innerHTML = '<div class="alert alert-danger">Erreur pendant la prédiction caméra.</div>';
        }
    } finally {
        isPredicting = false;
    }
}

async function capture360AndPredict(){
    const video = document.getElementById('cameraVideo');
    const resultBox = document.getElementById('cameraAiResult');

    if(!video || !video.srcObject){
        alert("Clique d'abord sur Open Camera.");
        return;
    }
    if(video.videoWidth === 0 || video.videoHeight === 0){
        alert("La caméra n'est pas encore prête.");
        return;
    }
    if(isPredicting){ return; }

    isPredicting = true;
    const formData = new FormData();
    const totalFrames = 8;

    try{
        for(let i=0; i<totalFrames; i++){
            if(resultBox){
                resultBox.innerHTML = `<div class="alert alert-info fw-bold">Inspection 360° : frame ${i+1}/${totalFrames}. Tourne la canette doucement.</div>`;
            }
            const blob = await captureVideoBlob(video, 0.92);
            formData.append('frames', blob, `view_${i+1}.jpg`);
            await sleep(650);
        }

        if(resultBox){
            resultBox.innerHTML = '<div class="text-muted fw-bold">Analyse 360° en cours...</div>';
        }

        const response = await fetch('/predict-360', { method:'POST', body:formData });
        const data = await response.json();

        if(data.error){
            if(resultBox){ resultBox.innerHTML = `<div class="alert alert-danger">${data.error}</div>`; }
            return;
        }

        updateDashboardFromPrediction(data);
        showCameraPredictionResult(data);
        refreshDashboard();

    }catch(err){
        console.error(err);
        if(resultBox){ resultBox.innerHTML = '<div class="alert alert-danger">Erreur pendant l inspection 360°.</div>'; }
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
                <div class="big-status ${statusClass}" style="font-size:1.25rem;padding:12px;">
                    ${data.quality_status}
                </div>
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
    fetch('/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action})}).then(r=>{ if(r.ok) document.getElementById('lastCommand').textContent=action; });
}
function setText(id, value){ const el=document.getElementById(id); if(el) el.textContent=value; }

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
        const qBox = document.getElementById('qualityStatusBox');
        qBox.textContent = c.quality_status;
        qBox.className = 'big-status ' + (c.quality_status === 'GOOD' ? 'good' : 'defect');
        qualityChart.data.labels = data.history.timestamps;
        qualityChart.data.datasets[0].data = data.history.good_rates;
        qualityChart.data.datasets[1].data = data.history.defect_rates;
        qualityChart.update('none');
        defectChart.data.datasets[0].data = [
    c.good_cans,
    c.defective_cans
];

defectChart.update('none');
        document.getElementById('historyTable').innerHTML = data.history.history_rows.slice().reverse().map(r => `<tr><td>${r.time}</td><td>${r.brand}</td><td>${r.result}</td></tr>`).join('');
    }catch(e){ console.error('Dashboard refresh error:', e); }
}
setInterval(refreshDashboard, 5000);
</script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    """, current=current_state, history=historical_data, history_rows=historical_data["history_rows"], defect_distribution=defect_distribution, brand_stats=brand_stats)


@app.route("/predict-image", methods=["POST"])
def predict_image():
    """Analyse d'une seule image. Attention: pour valider GOOD, utiliser /predict-360."""
    if "file" not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nom de fichier vide"}), 400

    if brand_model is None or defect_model is None:
        return jsonify({
            "error": "Modèles IA non chargés. Vérifie que les fichiers .keras sont dans le dossier models/."
        }), 500

    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}_{filename}"
    img_path = UPLOAD_FOLDER / filename
    file.save(img_path)

    try:
        result = predict_full_image(img_path)

        # Image simple = uniquement détection de marque / pré-analyse.
        # On ne valide jamais GOOD avec une seule frame.
        # Si le modèle dit GOOD, on affiche une instruction claire au lieu de GOOD.
        if result.get("quality_status") == "GOOD":
            result["quality_status"] = "INSPECT 360°"
            result["defect_type"] = "Appuyer sur Vérifier défectueuse ou pas"
            result["efficientnet_result"] = "En attente inspection 360°"
            result["patchcore_result"] = "Need 360°"
            result["raw_defect"] = "need_360"

        result["decision_rule"] = "Image simple: résultat indicatif. Utiliser Vérifier défectueuse ou pas."

        # Ne pas compter une image simple comme canette GOOD/DEFECT finale.
        # Le comptage final se fait après /predict-360.
        update_state_from_prediction(result, count_as_one=False)
    except Exception as e:
        return jsonify({"error": f"Erreur prédiction IA : {e}"}), 500

    result["current"] = current_state
    return jsonify(result)


@app.route("/predict-360", methods=["POST"])
def predict_360():
    """Analyse multi-images: la canette est validée GOOD seulement si toutes les vues sont GOOD."""
    files = request.files.getlist("frames")
    if not files:
        return jsonify({"error": "Aucune frame 360° envoyée"}), 400

    if brand_model is None or defect_model is None:
        return jsonify({
            "error": "Modèles IA non chargés. Vérifie que les fichiers .keras sont dans le dossier models/."
        }), 500

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    try:
        for i, file in enumerate(files, start=1):
            filename = secure_filename(file.filename or f"view_{i}.jpg")
            img_path = UPLOAD_FOLDER / f"{timestamp}_360_{i}_{filename}"
            file.save(img_path)
            results.append(predict_full_image(img_path))

        final_result = aggregate_360_results(results)
        update_state_from_prediction(final_result, count_as_one=True)

    except Exception as e:
        return jsonify({"error": f"Erreur prédiction 360° : {e}"}), 500

    final_result["current"] = current_state
    return jsonify(final_result)

@app.route("/quality-data")
def quality_data():
    return jsonify({"current": current_state, "history": historical_data})



@app.route("/control")
def control_panel():
    recalculate_percentages()
    batch_id = f"BATCH-{current_state['current_brand'].upper().replace(' ', '-')}-2026-001"
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CROWN MAGHREB - Control Center</title>

<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

<style>
:root{
    --green:#007a3d;
    --dark:#174d45;
    --teal:#2f7468;
    --orange:#d57435;
    --red:#c0392b;
    --border:#c8ddd8;
    --text:#203b36;
    --muted:#6b7d78;
    --panel:#ffffffee;
}

body{
    margin:0;
    min-height:100vh;
    background:
        linear-gradient(90deg, rgba(63,131,122,.08) 1px, transparent 1px),
        linear-gradient(rgba(63,131,122,.08) 1px, transparent 1px),
        linear-gradient(135deg,#eef6f2,#fff8df);
    background-size:34px 34px,34px 34px,auto;
    font-family:"Segoe UI",system-ui,sans-serif;
    color:var(--text);
}

.topbar{
    margin:16px;
    background:var(--panel);
    border:1px solid var(--border);
    border-radius:20px;
    padding:16px 22px;
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:16px;
    box-shadow:0 12px 28px rgba(28,92,83,.12);
}

.title{
    font-weight:950;
    font-size:1.35rem;
    color:var(--dark);
    letter-spacing:.2px;
}

.subtitle{
    color:var(--muted);
    font-size:.9rem;
}

.logo{
    width:132px;
    background:white;
    border-radius:14px;
    padding:10px;
}

.status-pill{
    background:#f4fbf7;
    border:1px solid var(--border);
    border-radius:999px;
    padding:8px 12px;
    font-size:.78rem;
    font-weight:850;
    white-space:nowrap;
}

.dot{
    width:9px;
    height:9px;
    display:inline-block;
    border-radius:50%;
    background:#17a75a;
    margin-right:6px;
    box-shadow:0 0 0 4px rgba(23,167,90,.12);
}

.card-control, .metric-card, .command-box{
    background:var(--panel);
    border:1px solid var(--border);
    border-radius:20px;
    box-shadow:0 12px 28px rgba(28,92,83,.12);
}

.card-control{
    padding:16px;
    height:100%;
    transition:.25s;
}

.card-control:hover{
    transform:translateY(-3px);
    box-shadow:0 18px 34px rgba(28,92,83,.18);
}

.metric-card{
    padding:16px;
    text-align:center;
    min-height:116px;
    background:linear-gradient(135deg,#f7fffb,#ffffff);
}

.metric-label{
    color:var(--muted);
    font-weight:850;
    font-size:.78rem;
    text-transform:uppercase;
}

.metric-value{
    font-size:1.75rem;
    font-weight:950;
    color:var(--green);
    line-height:1.05;
    margin-top:6px;
}

.metric-sub{
    color:#31544d;
    font-size:.82rem;
    margin-top:4px;
}

.card-title-custom{
    font-weight:950;
    color:var(--dark);
    margin-bottom:12px;
    display:flex;
    align-items:center;
    gap:10px;
    border-bottom:1px solid #e6eeee;
    padding-bottom:10px;
}

.icon-box{
    width:38px;
    height:38px;
    border-radius:12px;
    display:grid;
    place-items:center;
    background:#e6f4ef;
    color:var(--green);
}

.info-line{
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:12px;
    border-bottom:1px solid #e6eeee;
    padding:9px 0;
    font-weight:750;
}

.info-line span{
    color:var(--muted);
}

.btn-control{
    border:none;
    border-radius:14px;
    padding:11px 13px;
    font-weight:900;
    width:100%;
    margin-bottom:8px;
    transition:.18s;
}

.btn-control:hover{
    transform:scale(1.015);
    filter:brightness(1.02);
}

.btn-start{background:linear-gradient(135deg,#007a3d,#0b5f35);color:white;}
.btn-pause{background:linear-gradient(135deg,#f1c40f,#d4a20b);color:#203b36;}
.btn-stop{background:linear-gradient(135deg,#c0392b,#8e241b);color:white;}
.btn-dark-custom{background:linear-gradient(135deg,#174d45,#0e2925);color:white;}
.btn-orange{background:linear-gradient(135deg,#d57435,#a44a23);color:white;}
.btn-blue{background:linear-gradient(135deg,#2f80ed,#1b4f9c);color:white;}

.badge-online{
    background:#e7f7ef;
    color:#08783f;
    border:1px solid #bfe7d0;
    padding:6px 10px;
    border-radius:999px;
    font-weight:950;
    font-size:.72rem;
}

.badge-warning{
    background:#fff0e8;
    color:#b83222;
    border:1px solid #f0c5b4;
    padding:6px 10px;
    border-radius:999px;
    font-weight:950;
    font-size:.72rem;
}

.command-box{
    padding:16px;
}

.command-text{
    font-size:1.2rem;
    font-weight:950;
    color:var(--orange);
}

.log-box{
    max-height:170px;
    overflow:auto;
    font-size:.88rem;
}

.log-row{
    border-bottom:1px solid #e6eeee;
    padding:7px 0;
}

.progress{
    height:12px;
    border-radius:999px;
    background:#e6eeee;
}

.progress-bar{
    background:linear-gradient(90deg,#007a3d,#2f7468);
}

@media(max-width:900px){
    .topbar{flex-direction:column;align-items:flex-start;}
}
</style>
</head>

<body>

<div class="topbar">
    <div class="d-flex align-items-center gap-3 flex-wrap">
        <img src="/logo" class="logo" alt="CROWN logo">
        <div>
            <div class="title">CROWN MAGHREB - DECO MACHINE CONTROL CENTER</div>
            <div class="subtitle">Industrial AI inspection control panel · Brand recognition · Defect detection</div>
        </div>
    </div>

    <div class="d-flex gap-2 flex-wrap align-items-center justify-content-end">
        <span class="status-pill"><span class="dot"></span>Camera Connected</span>
        <span class="status-pill"><span class="dot"></span>AI Online</span>
        <span class="status-pill"><span class="dot"></span>Raspberry Pi Connected</span>
        <a href="/" class="btn btn-dark-custom btn-sm">Dashboard</a>
    </div>
</div>

<div class="container-fluid px-4 pb-4">

<div class="row g-3 mb-3">
    <div class="col-md-3">
        <div class="metric-card">
            <div class="metric-label">Current Brand</div>
            <div class="metric-value" id="brandValue">{{ current.current_brand }}</div>
            <div class="metric-sub">Brand confidence: <span id="brandConfidence">{{ current.brand_confidence }}</span>%</div>
        </div>
    </div>

    <div class="col-md-3">
        <div class="metric-card">
            <div class="metric-label">Quality Status</div>
            <div class="metric-value" id="qualityValue">{{ current.quality_status }}</div>
            <div class="metric-sub">Current defect: <strong id="defectTypeValue">{{ current.defect_type }}</strong></div>
        </div>
    </div>

    <div class="col-md-3">
        <div class="metric-card">
            <div class="metric-label">Inspection Mode</div>
            <div class="metric-value" id="modeValue">AUTO</div>
            <div class="metric-sub">Deco Machine Line</div>
        </div>
    </div>

    <div class="col-md-3">
        <div class="metric-card">
            <div class="metric-label">Batch ID</div>
            <div class="metric-value" style="font-size:1.25rem;" id="batchValue">{{ batch_id }}</div>
            <div class="metric-sub">Dynamic production batch</div>
        </div>
    </div>
</div>

<div class="row g-3 mb-3">
    <div class="col-md-3">
        <div class="metric-card">
            <div class="metric-label">Total Inspected</div>
            <div class="metric-value" id="totalInspected">{{ current.total_inspected }}</div>
            <div class="metric-sub">All cans analyzed</div>
        </div>
    </div>

    <div class="col-md-3">
        <div class="metric-card">
            <div class="metric-label">Accepted</div>
            <div class="metric-value" id="goodCans">{{ current.good_cans }}</div>
            <div class="metric-sub"><span id="acceptedPercent">{{ current.accepted_percent }}</span>% accepted</div>
            <div class="progress mt-2"><div class="progress-bar" id="acceptedBar" style="width: {{ current.accepted_percent }}%;"></div></div>
        </div>
    </div>

    <div class="col-md-3">
        <div class="metric-card">
            <div class="metric-label">Rejected</div>
            <div class="metric-value" style="color:var(--orange);" id="defectiveCans">{{ current.defective_cans }}</div>
            <div class="metric-sub"><span id="defectRate">{{ current.defect_rate }}</span>% defect rate</div>
        </div>
    </div>

    <div class="col-md-3">
        <div class="metric-card">
            <div class="metric-label">Last Inspection</div>
            <div class="metric-value" style="font-size:1.1rem;" id="lastInspection">{{ current.timestamp }}</div>
            <div class="metric-sub">Live dashboard synchronization</div>
        </div>
    </div>
</div>

<div class="row g-3">

<div class="col-lg-4">
<div class="card-control">
    <div class="card-title-custom">
        <div class="icon-box"><i class="fa-solid fa-play"></i></div>
        Inspection Controls
    </div>

    <button onclick="sendAction('start_inspection')" class="btn-control btn-start">
        <i class="fa-solid fa-play me-2"></i>Start Inspection
    </button>

    <button onclick="sendAction('pause_inspection')" class="btn-control btn-pause">
        <i class="fa-solid fa-pause me-2"></i>Pause Inspection
    </button>

    <button onclick="sendAction('stop_inspection')" class="btn-control btn-stop">
        <i class="fa-solid fa-stop me-2"></i>Stop Inspection
    </button>

    <button onclick="sendAction('reset_counters')" class="btn-control btn-dark-custom">
        <i class="fa-solid fa-rotate-left me-2"></i>Reset Counters
    </button>
</div>
</div>

<div class="col-lg-4">
<div class="card-control">
    <div class="card-title-custom">
        <div class="icon-box"><i class="fa-solid fa-robot"></i></div>
        AI Models Control
    </div>

    <div class="info-line"><span>Brand Classifier</span><strong class="badge-online">ONLINE</strong></div>
    <div class="info-line"><span>PatchCore</span><strong id="patchcoreBadge" class="badge-online">{{ current.patchcore_result }}</strong></div>
    <div class="info-line"><span>EfficientNetB0</span><strong id="efficientnetBadge" class="badge-online">{{ current.efficientnet_result }}</strong></div>

    <button onclick="sendAction('reload_models')" class="btn-control btn-blue mt-3">
        <i class="fa-solid fa-brain me-2"></i>Reload AI Models
    </button>
</div>
</div>

<div class="col-lg-4">
<div class="card-control">
    <div class="card-title-custom">
        <div class="icon-box"><i class="fa-solid fa-microchip"></i></div>
        Raspberry Pi & Camera
    </div>

    <div class="info-line"><span>Camera</span><strong class="badge-online">{{ current.camera_status }}</strong></div>
    <div class="info-line"><span>Raspberry Pi</span><strong class="badge-online">{{ current.raspberry_status }}</strong></div>
    <div class="info-line"><span>Database</span><strong class="badge-online">CONNECTED</strong></div>

    <button onclick="sendAction('test_camera')" class="btn-control btn-blue mt-3">
        <i class="fa-solid fa-camera me-2"></i>Test Camera
    </button>

    <button onclick="sendAction('restart_raspberry')" class="btn-control btn-dark-custom">
        <i class="fa-solid fa-wifi me-2"></i>Restart Raspberry Link
    </button>
</div>
</div>

<div class="col-lg-6">
<div class="card-control">
    <div class="card-title-custom">
        <div class="icon-box"><i class="fa-solid fa-flask"></i></div>
        Manual Defect Simulation
    </div>

    <button onclick="sendAction('simulate_good')" class="btn-control btn-start">
        <i class="fa-solid fa-circle-check me-2"></i>Simulate GOOD Can
    </button>

    <button onclick="sendAction('simulate_missing_paint')" class="btn-control btn-stop">
        <i class="fa-solid fa-paint-roller me-2"></i>Simulate Missing Paint
    </button>

    <button onclick="sendAction('simulate_contamination')" class="btn-control btn-orange">
        <i class="fa-solid fa-droplet me-2"></i>Simulate Color Contamination
    </button>
</div>
</div>

<div class="col-lg-6">
<div class="card-control">
    <div class="card-title-custom">
        <div class="icon-box"><i class="fa-solid fa-gears"></i></div>
        Machine Actions
    </div>

    <button onclick="sendAction('green_led')" class="btn-control btn-start">
        <i class="fa-solid fa-lightbulb me-2"></i>Green LED / GOOD Signal
    </button>

    <button onclick="sendAction('red_led')" class="btn-control btn-stop">
        <i class="fa-solid fa-triangle-exclamation me-2"></i>Red LED / DEFECT Signal
    </button>

    <button onclick="sendAction('buzzer')" class="btn-control btn-orange">
        <i class="fa-solid fa-bell me-2"></i>Buzzer Alert
    </button>

    <button onclick="sendAction('reject_can')" class="btn-control btn-dark-custom">
        <i class="fa-solid fa-ban me-2"></i>Reject Can Signal
    </button>
</div>
</div>

</div>

<div class="row g-3 mt-1">
    <div class="col-lg-5">
        <div class="command-box">
            <div class="text-muted fw-bold">LAST COMMAND SENT TO SYSTEM</div>
            <div id="lastCommand" class="command-text">None</div>
        </div>
    </div>

    <div class="col-lg-7">
        <div class="command-box">
            <div class="text-muted fw-bold mb-2">SYSTEM LOG</div>
            <div id="systemLog" class="log-box">
                <div class="log-row">System initialized and synchronized with dashboard</div>
            </div>
        </div>
    </div>
</div>

</div>

<script>
function addLog(message){
    const log = document.getElementById('systemLog');
    const now = new Date().toLocaleTimeString();
    const row = document.createElement('div');
    row.className = 'log-row';
    row.textContent = now + ' - ' + message;
    log.prepend(row);
}

function setText(id, value){
    const el = document.getElementById(id);
    if(el){ el.textContent = value; }
}

function setQualityStyle(status){
    const q = document.getElementById('qualityValue');
    if(!q) return;
    if(status === 'DEFECT'){
        q.style.color = '#c0392b';
    }else{
        q.style.color = '#007a3d';
    }
}

function updateBatch(brand){
    const safeBrand = String(brand).toUpperCase().replaceAll(' ', '-');
    setText('batchValue', 'BATCH-' + safeBrand + '-2026-001');
}

function updateLiveCards(action){
    if(action === 'start_inspection'){
        setText('modeValue', 'AUTO');
        addLog('Inspection started');
    }

    if(action === 'pause_inspection'){
        setText('modeValue', 'PAUSE');
        addLog('Inspection paused');
    }

    if(action === 'stop_inspection'){
        setText('modeValue', 'STOP');
        addLog('Inspection stopped');
    }

    if(action === 'simulate_good'){
        setText('qualityValue', 'GOOD');
        setText('defectTypeValue', 'None');
        setText('patchcoreBadge', 'Normal');
        setText('efficientnetBadge', 'Good');
        setQualityStyle('GOOD');
        addLog('GOOD can simulated');
    }

    if(action === 'simulate_missing_paint'){
        setText('qualityValue', 'DEFECT');
        setText('defectTypeValue', 'Missing Paint');
        setText('patchcoreBadge', 'Anomaly');
        setText('efficientnetBadge', 'Missing Paint');
        setQualityStyle('DEFECT');
        addLog('Missing Paint defect simulated');
    }

    if(action === 'simulate_contamination'){
        setText('qualityValue', 'DEFECT');
        setText('defectTypeValue', 'Color Contamination');
        setText('patchcoreBadge', 'Anomaly');
        setText('efficientnetBadge', 'Color Contamination');
        setQualityStyle('DEFECT');
        addLog('Color Contamination defect simulated');
    }

    if(action === 'reload_models'){ addLog('AI models reloaded'); }
    if(action === 'test_camera'){ addLog('Camera test command sent'); }
    if(action === 'restart_raspberry'){ addLog('Raspberry connection restart command sent'); }
    if(action === 'green_led'){ addLog('Green LED signal sent'); }
    if(action === 'red_led'){ addLog('Red LED signal sent'); }
    if(action === 'buzzer'){ addLog('Buzzer alert sent'); }
    if(action === 'reject_can'){ addLog('Reject can signal sent'); }
    if(action === 'reset_counters'){ addLog('Production counters reset'); }
}

async function refreshControlData(){
    try{
        const response = await fetch('/quality-data');
        if(!response.ok) return;
        const data = await response.json();
        const c = data.current;

        setText('brandValue', c.current_brand);
        setText('brandConfidence', c.brand_confidence);
        setText('qualityValue', c.quality_status);
        setText('defectTypeValue', c.defect_type);
        setText('totalInspected', c.total_inspected);
        setText('goodCans', c.good_cans);
        setText('defectiveCans', c.defective_cans);
        setText('acceptedPercent', c.accepted_percent);
        setText('defectRate', c.defect_rate);
        setText('lastInspection', c.timestamp);
        setText('patchcoreBadge', c.patchcore_result);
        setText('efficientnetBadge', c.efficientnet_result);

        const acceptedBar = document.getElementById('acceptedBar');
        if(acceptedBar){ acceptedBar.style.width = Number(c.accepted_percent) + '%'; }

        updateBatch(c.current_brand);
        setQualityStyle(c.quality_status);
    }catch(e){
        console.error('Control refresh error:', e);
    }
}

function sendAction(action){
    fetch('/action',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({action:action})
    })
    .then(response=>response.json())
    .then(data=>{
        document.getElementById('lastCommand').textContent = action;
        updateLiveCards(action);
        refreshControlData();
    })
    .catch(error=>{
        addLog('Command error: ' + action);
        console.error(error);
    });
}

refreshControlData();
setInterval(refreshControlData, 5000);
</script>

</body>
</html>
""", current=current_state, batch_id=batch_id)


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
        current_state["line_status"] = "Paused"

    elif action == "stop_inspection":
        current_state["line_status"] = "Stopped"

    recalculate_percentages()

    if action in ["simulate_good", "simulate_missing_paint", "simulate_contamination", "reset_counters"]:
        historical_data["timestamps"].append(current_state["timestamp"])
        historical_data["good_rates"].append(current_state["accepted_percent"])
        historical_data["defect_rates"].append(current_state["defect_rate"])
        historical_data["history_rows"].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "brand": current_state["current_brand"],
            "result": current_state["defect_type"] if current_state["quality_status"] == "DEFECT" else "Good",
        })

        max_points = 50
        for key in historical_data:
            historical_data[key] = historical_data[key][-max_points:]

    if action:
        send_to_server(action)

    return jsonify({"status": "OK", "action": action, "current": current_state})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
