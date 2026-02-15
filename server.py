import json
import os
import random
import string
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import requests
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="public")

PORT = 3000
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data.json"
ARCHIVE_DIR = BASE_DIR / "archives"

MODEL_A = "llama3.2:3b"
MODEL_B = "gemma2:2b"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


def ensure_archives_dir():
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def archive_current_data():
    if not DATA_FILE.exists():
        return
    ensure_archives_dir()
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    archive_path = ARCHIVE_DIR / f"data_{timestamp}.json"
    DATA_FILE.rename(archive_path)
    print("Previous run archived to", archive_path)


def init_new_run():
    archive_current_data()
    data = {
        "modelA": MODEL_A,
        "modelB": MODEL_B,
        "rounds": [],
    }
    DATA_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def load_data():
    if DATA_FILE.exists():
        return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    return init_new_run()


def save_data(data):
    DATA_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_with_ollama(model: str, prompt: str) -> tuple[str, float]:
    start = time.perf_counter()
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=300,
    )
    r.raise_for_status()
    out = r.json()
    elapsed = time.perf_counter() - start
    return out.get("response", ""), elapsed


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/data")
def api_data():
    data = load_data()
    return jsonify({
        "modelA": data["modelA"],
        "modelB": data["modelB"],
        "rounds": [
            {"id": r["id"], "prompt": r["prompt"], "vote": r.get("vote")}
            for r in data["rounds"]
        ],
    })


@app.route("/api/archive", methods=["POST"])
def api_archive():
    """Перенести текущий data.json в архив и создать новый пустой."""
    init_new_run()
    return jsonify({"ok": True})


@app.route("/api/scores")
def api_scores():
    data = load_data()
    score_a = score_b = 0
    for r in data["rounds"]:
        if r.get("vote") == 1:
            if r["model1"] == data["modelA"]:
                score_a += 1
            else:
                score_b += 1
        elif r.get("vote") == 2:
            if r["model2"] == data["modelA"]:
                score_a += 1
            else:
                score_b += 1
    return jsonify({
        "modelA": data["modelA"],
        "modelB": data["modelB"],
        "scoreA": score_a,
        "scoreB": score_b,
        "totalRounds": sum(1 for r in data["rounds"] if r.get("vote") is not None),
    })


@app.route("/api/round", methods=["POST"])
def api_round():
    body = request.get_json() or {}
    prompt = body.get("prompt")
    if not prompt or not isinstance(prompt, str):
        return jsonify({"error": "prompt required"}), 400
    prompt = prompt.strip()
    data = load_data()
    try:
        with ThreadPoolExecutor(2) as ex:
            f_a = ex.submit(generate_with_ollama, MODEL_A, prompt)
            f_b = ex.submit(generate_with_ollama, MODEL_B, prompt)
            resp_a, time_a = f_a.result()
            resp_b, time_b = f_b.result()
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500
    swap = random.random() < 0.5
    answer1 = resp_b if swap else resp_a
    answer2 = resp_a if swap else resp_b
    time1 = time_b if swap else time_a
    time2 = time_a if swap else time_b
    model1 = MODEL_B if swap else MODEL_A
    model2 = MODEL_A if swap else MODEL_B
    round_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
    round_ = {
        "id": round_id,
        "prompt": prompt,
        "answer1": answer1,
        "answer2": answer2,
        "time1": round(time1, 2),
        "time2": round(time2, 2),
        "model1": model1,
        "model2": model2,
        "vote": None,
    }
    data["rounds"].append(round_)
    save_data(data)
    return jsonify({
        "roundId": round_id,
        "answer1": answer1,
        "answer2": answer2,
        "time1": round(time1, 2),
        "time2": round(time2, 2),
    })


@app.route("/api/vote", methods=["POST"])
def api_vote():
    body = request.get_json() or {}
    round_id = body.get("roundId")
    vote = body.get("vote")
    if round_id is None or vote not in (1, 2):
        return jsonify({"error": "roundId and vote (1 or 2) required"}), 400
    data = load_data()
    round_ = next((r for r in data["rounds"] if r["id"] == round_id), None)
    if not round_:
        return jsonify({"error": "round not found"}), 404
    round_["vote"] = vote
    chosen_model = round_["model1"] if vote == 1 else round_["model2"]
    save_data(data)
    score_a = score_b = 0
    for r in data["rounds"]:
        if r.get("vote") == 1:
            if r["model1"] == data["modelA"]:
                score_a += 1
            else:
                score_b += 1
        elif r.get("vote") == 2:
            if r["model2"] == data["modelA"]:
                score_a += 1
            else:
                score_b += 1
    return jsonify({
        "chosenModel": chosen_model,
        "scores": {data["modelA"]: score_a, data["modelB"]: score_b},
        "totalRounds": sum(1 for r in data["rounds"] if r.get("vote") is not None),
    })


if __name__ == "__main__":
    init_new_run()
    print(f"Ollama LM Arena at http://localhost:{PORT}")
    print(f"Models: {MODEL_A} vs {MODEL_B}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
