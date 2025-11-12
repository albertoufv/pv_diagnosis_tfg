"""
Utilidades: logging, guardar/cargar, visualización.
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np

def create_directories():
    """Crea carpetas necesarias si no existen."""
    dirs = ['data/raw', 'data/processed', 'models', 'logs', 'results', 'notebooks']
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def log_experiment(experiment_name, metrics, params):
    """Guarda logs de experimento."""
    log_data = {
        'name': experiment_name,
        'metrics': metrics,
        'params': params
    }
    with open(f'logs/{experiment_name}.json', 'w') as f:
        json.dump(log_data, f, indent=2)

def visualize_thermal_image(img, title="Imagen Térmica"):
    """Visualiza imagen térmica con escala de colores."""
    plt.figure(figsize=(8, 6))
    plt.imshow(img, cmap='hot')
    plt.colorbar(label='Temperatura (normalizada)')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    return plt

def plot_timeseries(data, title="Serie Temporal SCADA"):
    """Dibuja serie temporal."""
    plt.figure(figsize=(12, 4))
    plt.plot(data)
    plt.title(title)
    plt.xlabel('Tiempo')
    plt.ylabel('Valor')
    plt.grid(True, alpha=0.3)
    return plt