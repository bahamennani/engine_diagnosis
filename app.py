
from flask import Flask, request, jsonify
import os
import numpy as np
import tensorflow as tf
import librosa

app = Flask(__name__)

def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def predict(audio_path, model_path):
    interpreter, input_details, output_details = load_model(model_path)
    y, sr = librosa.load(audio_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc = mfcc.T[:50]
    mfcc = np.expand_dims(mfcc, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], mfcc)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data)
    return int(prediction)

@app.route("/predict", methods=["POST"])
def predict_audio():
    if 'file' not in request.files or 'engine_type' not in request.form:
        return jsonify({"error": "Missing file or engine_type"}), 400

    audio = request.files['file']
    engine_type = request.form['engine_type']

    if engine_type not in ["essence", "diesel"]:
        return jsonify({"error": "Invalid engine_type"}), 400

    save_path = "temp.wav"
    audio.save(save_path)

    model_path = f"models/cnn_model_{engine_type}.tflite"
    result = predict(save_path, model_path)
    os.remove(save_path)

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
