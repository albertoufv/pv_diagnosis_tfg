"""
Definición de modelos: CNN, LSTM, Fusión Multimodal, Ensemble.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

class ThermalCNN:
    """CNN con transfer learning (ResNet50) para imágenes térmicas."""
    
    def build(self, input_shape=(224, 224, 1)):
        """
        - Carga ResNet50 preentrenada en ImageNet.
        - Congela capas iniciales, entrena solo las finales.
        - Output: embedding de 128 dimensiones.
        """
        # Cargar ResNet50 preentrenado
        base_model = keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Congelar capas
        base_model.trainable = False
        
        # Añadir cabeza personalizada
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),  # Embedding de 128 dims
        ])
        
        return model

class ScadaLSTM:
    """LSTM bidireccional para series temporales SCADA."""
    
    def build(self, input_shape=(24, 6)):  # 24 pasos, 6 features (V, I, P, T, etc)
        """
        - Procesa secuencias de 24 horas.
        - LSTM bidireccional para captar patrones pasados y futuros.
        - Output: embedding de 128 dimensiones.
        """
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Dropout(0.3),
            layers.Bidirectional(layers.LSTM(32, return_sequences=False)),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),  # Embedding de 128 dims
        ])
        
        return model

class MultimodalFusion:
    """Fusión de embeddings CNN + LSTM con Attention."""
    
    def build(self, thermal_embedding_dim=128, scada_embedding_dim=128, num_classes=2):
        """
        - Recibe embeddings de CNN (128d) y LSTM (128d).
        - Aplica capas Dense, concatena, predice anomalía (0=normal, 1=fallo).
        - Con mecanismo Attention para dar peso a qué modal importa más.
        """
        # Entrada térmica
        thermal_input = keras.Input(shape=(thermal_embedding_dim,), name='thermal')
        # Entrada SCADA
        scada_input = keras.Input(shape=(scada_embedding_dim,), name='scada')
        
        # Procesamiento individual
        thermal_processed = layers.Dense(64, activation='relu')(thermal_input)
        scada_processed = layers.Dense(64, activation='relu')(scada_input)
        
        # Concatenación
        concatenated = layers.Concatenate()([thermal_processed, scada_processed])
        
        # Capas adicionales
        fusion = layers.Dense(128, activation='relu')(concatenated)
        fusion = layers.Dropout(0.4)(fusion)
        
        # Output (logits para 2 clases)
        output = layers.Dense(num_classes, activation='softmax')(fusion)
        
        model = models.Model(inputs=[thermal_input, scada_input], outputs=output)
        return model

class EnsembleClassifier:
    """Ensamble de modelos clásicos para máxima robustez."""
    
    def __init__(self):
        """Crea 4 modelos clásicos + vota."""
        self.rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        self.svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
        self.xgb = XGBClassifier(n_estimators=200, max_depth=10, random_state=42)
        self.mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    
    def fit(self, X, y):
        """Entrena los 4 modelos."""
        self.rf.fit(X, y)
        self.svm.fit(X, y)
        self.xgb.fit(X, y)
        self.mlp.fit(X, y)
    
    def predict(self, X):
        """Vota: mayoría de predicciones."""
        rf_pred = self.rf.predict(X)
        svm_pred = self.svm.predict(X)
        xgb_pred = self.xgb.predict(X)
        mlp_pred = self.mlp.predict(X)
        
        # Votación mayoritaria
        votes = rf_pred + svm_pred + xgb_pred + mlp_pred
        return (votes >= 2).astype(int)  # Si ≥2 votos para 1, es anomalía
    
    def predict_proba(self, X):
        """Promedia probabilidades de los 4 modelos."""
        proba_avg = (
            self.rf.predict_proba(X) +
            self.svm.predict_proba(X) +
            self.xgb.predict_proba(X) +
            self.mlp.predict_proba(X)
        ) / 4
        return proba_avg
