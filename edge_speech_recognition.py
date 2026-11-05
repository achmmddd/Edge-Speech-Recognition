"""
Modul Program Python Lokal: Edge Speech-to-Text Inference
Proyek: Edge Speech-to-Text menggunakan Whisper dan ONNX Runtime
Deskripsi: Skrip ini digunakan untuk melakukan preprocessing audio, ekstraksi fitur log-mel spectrogram, 
           dan inferensi model ASR secara real-time pada perangkat edge (seperti Raspberry Pi 4) 
           menggunakan ONNX Runtime.
"""

import numpy as np
import librosa
import onnxruntime as ort
from transformers import WhisperFeatureExtractor

class EdgeASRInference:
    def __init__(self, encoder_onnx_path: str, model_name: str = "openai/whisper-tiny"):
        """
        Inisialisasi sesi ONNX Runtime dan Feature Extractor untuk Whisper pada perangkat edge.
        """
        print("Inisialisasi Edge ASR Inference Engine...")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        
        # Load ONNX Runtime session dengan optimasi execution provider (CPU / Execution providers yang tersedia)
        providers = ['CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(encoder_onnx_path, providers=providers)
        print(f"Model ONNX berhasil dimuat dari: {encoder_onnx_path}")

    def preprocess_audio(self, audio_path: str, target_sr: int = 16000) -> np.ndarray:
        """
        Melakukan resampling audio ke 16kHz mono dan normalisasi.
        """
        print(f"Memuat dan memproses audio dari: {audio_path}")
        y, sr = librosa.load(audio_path, sr=target_sr)
        
        # Ekstraksi fitur log-mel spectrogram menggunakan WhisperFeatureExtractor
        inputs = self.feature_extractor(
            y, 
            sampling_rate=target_sr, 
            return_tensors="np"
        )
        input_features = inputs.input_features
        return input_features

    def run_inference(self, input_features: np.ndarray) -> np.ndarray:
        """
        Menjalankan inferensi encoder menggunakan ONNX Runtime.
        """
        input_name = self.ort_session.get_inputs()[0].name
        ort_inputs = {input_name: input_features}
        
        # Eksekusi inferensi pada edge device
        ort_outs = self.ort_session.run(None, ort_inputs)
        return ort_outs[0]

if __name__ == "__main__":
    # Contoh Penggunaan (Contoh Path Model dan Audio)
    encoder_path = "whisper_tiny_encoder.onnx"
    sample_audio = "sample_speech.wav"
    
    print("=== PROGRAM UTAMA EDGE SPEECH-TO-TEXT ===")
    print("Pastikan file model ONNX dan file audio tersedia sebelum menjalankan inferensi.")
    
    # Inisialisasi (Contoh simulasi struktur kode lokal)
    # asr_engine = EdgeASRInference(encoder_path)
    # features = asr_engine.preprocess_audio(sample_audio)
    # outputs = asr_engine.run_inference(features)
    # print("Hasil inferensi encoder berhasil diekstrak:", outputs.shape)
