"""
Entrenamiento de todos los modelos.
"""
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pickle

class ModelTrainer:
    
    def train_deep_learning(self, thermal_images, scada_sequences, labels, epochs=50):
        """
        1. Entrena CNN con imágenes térmicas.
        2. Entrena LSTM con series SCADA.
        3. Entrena fusión multimodal.
        4. Guarda todos los modelos.
        """
        from models import ThermalCNN, ScadaLSTM, MultimodalFusion
        
        # Split datos
        X_train, X_test, y_train, y_test = train_test_split(
            thermal_images, labels, test_size=0.2, random_state=42
        )
        
        # Entrenar CNN
        cnn = ThermalCNN().build()
        cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        cnn.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2)
        cnn.save('models/cnn_thermal.h5')
        
        # Entrenar LSTM
        lstm = ScadaLSTM().build()
        lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        lstm.fit(scada_sequences[:len(X_train)], y_train, epochs=epochs, batch_size=32)
        lstm.save('models/lstm_scada.h5')
        
        # Extraer embeddings para fusion
        cnn_embeddings = cnn.predict(X_train)
        lstm_embeddings = lstm.predict(scada_sequences[:len(X_train)])
        
        # Entrenar fusión
        fusion = MultimodalFusion().build()
        fusion.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        fusion.fit(
            [cnn_embeddings, lstm_embeddings], y_train,
            epochs=epochs, batch_size=32
        )
        fusion.save('models/fusion_multimodal.h5')
        
        print("✓ Modelos deep learning entrenados y guardados.")
    
    def train_ensemble(self, embeddings, labels):
        """
        Entrena el ensamble (RF + SVM + XGBoost + MLP) con embeddings fusionados.
        """
        from models import EnsembleClassifier
        
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42
        )
        
        ensemble = EnsembleClassifier()
        ensemble.fit(X_train, y_train)
        
        # Evalúa
        y_pred = ensemble.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"✓ Ensamble entrenado. Accuracy: {acc:.3f}, F1: {f1:.3f}")
        
        # Guarda el ensamble
        with open('models/ensemble.pkl', 'wb') as f:
            pickle.dump(ensemble, f)

# USO:
# trainer = ModelTrainer()
# trainer.train_deep_learning(thermal_images, scada_sequences, labels, epochs=50)
# trainer.train_ensemble(embeddings, labels)
