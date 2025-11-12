"""
Inferencia: carga modelos y predice en nuevas imágenes/datos SCADA.
"""
import tensorflow as tf
import pickle
import numpy as np

class DiagnosisPredictor:
    
    def __init__(self):
        """Carga todos los modelos entrenados."""
        self.cnn = tf.keras.models.load_model('models/cnn_thermal.h5')
        self.lstm = tf.keras.models.load_model('models/lstm_scada.h5')
        self.fusion = tf.keras.models.load_model('models/fusion_multimodal.h5')
        
        with open('models/ensemble.pkl', 'rb') as f:
            self.ensemble = pickle.load(f)
    
    def predict(self, thermal_image, scada_sequence):
        """
        Input:
          - thermal_image: (224, 224, 1) imagen normalizada
          - scada_sequence: (24, 6) secuencia SCADA normalizada
        
        Output:
          - diagnosis: 'NORMAL' o 'ANOMALÍA DETECTADA'
          - confidence: 0.0-1.0
          - explanation: qué contribuyó más (térmica o SCADA)
        """
        # Obtener embeddings
        cnn_embedding = self.cnn.predict(np.expand_dims(thermal_image, 0))[0]
        lstm_embedding = self.lstm.predict(np.expand_dims(scada_sequence, 0))[0]
        
        # Fusión multimodal
        fusion_embedding = self.fusion.predict([
            np.expand_dims(cnn_embedding, 0),
            np.expand_dims(lstm_embedding, 0)
        ])[0]
        
        # Predicción ensamble
        ensemble_pred = self.ensemble.predict_proba(np.expand_dims(fusion_embedding, 0))[0]
        
        confidence = ensemble_pred[1]  # Prob de anomalía
        diagnosis = 'ANOMALÍA DETECTADA' if confidence > 0.5 else 'NORMAL'
        
        return {
            'diagnosis': diagnosis,
            'confidence': confidence,
            'cnn_embedding': cnn_embedding,
            'lstm_embedding': lstm_embedding
        }

# USO:
# predictor = DiagnosisPredictor()
# result = predictor.predict(thermal_image, scada_sequence)
# print(f"{result['diagnosis']} (confianza: {result['confidence']:.2%})")
