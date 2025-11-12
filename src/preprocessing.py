"""
Funciones de preprocesamiento de datos térmicos y SCADA.
"""
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class ThermalPreprocessor:
    """Procesa imágenes térmicas."""
    
    def load_thermal_image(self, path):
        """Carga imagen térmica (TIF/PNG) y normaliza valores."""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # Normaliza a rango 0-1
        img_normalized = (img - img.min()) / (img.max() - img.min())
        return img_normalized
    
    def extract_hotspots(self, img, threshold=0.8):
        """Identifica zonas calientes (hot spots) en la imagen."""
        hotspot_mask = img > threshold
        hotspots = np.where(hotspot_mask)
        return hotspots, img[hotspot_mask]
    
    def augment_thermal_image(self, img):
        """Aumenta datos: rotaciones, flip, pequeños ruidos."""
        # Rotación aleatoria
        angle = np.random.randint(-15, 15)
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        
        # Flip horizontal
        flipped = cv2.flip(rotated, 1)
        return flipped

class ScadaPreprocessor:
    """Procesa datos SCADA (series temporales eléctricas)."""
    
    def load_scada_data(self, csv_path):
        """Carga CSV SCADA (Voltage, Current, Power, Temperature, etc.)."""
        df = pd.read_csv(csv_path)
        return df
    
    def resample_timeseries(self, df, freq='1H'):
        """Resamplea datos a frecuencia consistente (ej: cada hora)."""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_resampled = df.set_index('timestamp').resample(freq).mean()
        return df_resampled
    
    def normalize_features(self, df):
        """Normaliza cada columna (Voltage, Current, Power) a media=0, std=1."""
        scaler = StandardScaler()
        df_normalized = pd.DataFrame(
            scaler.fit_transform(df),
            columns=df.columns,
            index=df.index
        )
        return df_normalized
    
    def create_sequences(self, data, lookback=24):
        """Crea secuencias para LSTM (ej: últimas 24 horas predice siguiente 1)."""
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:i+lookback])
            y.append(data[i+lookback])
        return np.array(X), np.array(y)

# USO:
# thermal_prep = ThermalPreprocessor()
# img = thermal_prep.load_thermal_image('data/raw/imagen_001.tif')
# hotspots, temps = thermal_prep.extract_hotspots(img)

# scada_prep = ScadaPreprocessor()
# df = scada_prep.load_scada_data('data/raw/scada.csv')
# df_norm = scada_prep.normalize_features(df)
# X, y = scada_prep.create_sequences(df_norm, lookback=24)
