import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import PIL
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import PIL.Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121, DenseNet201 , DenseNet264
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Resize,
    EnsureType, 
    Lambda,
    Compose,
    LoadImage,
    RepeatChannel,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)
from monai.utils import set_determinism

print_config()

# Define la ruta de tu dataset
root_dir = "/"  # Cambiar esta ruta en caso necesario 
image_base_dir = "/CXR-TB/RESULTS/nmessino/DEV/pTB_LungRegionExtractor/RESULTS/resultados_nuevos/cropped_clahe" # Cambiar esta ruta en caso necesario
# Verifica que la ruta existe
if not os.path.exists(root_dir):
    raise FileNotFoundError(f"La ruta {root_dir} no existe. Verifica la ubicación de tu dataset.")

print(f"Directorio del dataset: {root_dir}")

set_determinism(seed=0) 

# OBTENER EL CSV
# Ruta al archivo CSV
csv_path = os.path.join(os.path.dirname(__file__), "dataset_ptbred_cism_splits_paper_ncomm.csv")

def file_exists(path):
    return isinstance(path, str) and os.path.exists(path) and os.path.isfile(path)

# --- Cargar y filtrar el dataset ---
df = pd.read_csv(csv_path)

# Filtrar los casos válidos
valid_cases = ["confirmed", "possible", "probable", "ltbi", "control", "unlikely"]#revisar porque puede que no haga falta
df = df[df["tuberculosis_type"].isin(valid_cases)]

# Verificar si las imágenes existen
df["exists_AP"] = df["filepath_AP"].apply(file_exists)
df["exists_LAT"] = df["filepath_LAT"].apply(file_exists)

# Filtrar: nos quedamos con filas donde al menos una imagen existe
df = df[df["exists_AP"] | df["exists_LAT"]]

# --- Crear el diccionario de etiquetas ---
label_dict = df[df["exists_AP"]].set_index("filepath_AP")[["TB_suggestive_minority", "tuberculosis_type"]].to_dict(orient="index")
df_lat_unique = df[df["exists_LAT"]].drop_duplicates(subset="filepath_LAT")
label_dict.update(df_lat_unique.set_index("filepath_LAT")[["TB_suggestive_minority", "tuberculosis_type"]].to_dict(orient="index"))

# --- Obtener listas finales de imágenes ---
image_files_AP = df[df["exists_AP"]]["filepath_AP"].tolist()
image_files_LAT = df[df["exists_LAT"]]["filepath_LAT"].tolist()

image_files_list = image_files_AP + image_files_LAT
image_class = [0] * len(image_files_AP) + [1] * len(image_files_LAT)

# Contar imágenes
num_total = len(image_class)

# Comprobación
print("Folds disponibles:", df["fold_cv"].unique())
print(f"Total de imágenes válidas: {len(image_files_list)}")
print(f"Ejemplo de ruta: {image_files_list[0]}")


# Hasta ahora hemos creado la configuracion necesaria (imports) , hemos establecido la ruta correcta al dataset y comprobado que puede acceder correctamente a las imagenes.
# Tambien hemos creado un dataset con las columnas que nos interesan , a partir de ahi hemos creado unos diccionarios con las etiquetas que comprobaremos si esta correcto o no.


#PREPARACION DE DATOS 
def get_labels(image_path):
    labels = label_dict.get(image_path, {"tuberculosis_type": "unknown", "TB_suggestive_minority": 0})
    tb_type = labels["tuberculosis_type"]
    minority = labels["TB_suggestive_minority"]

    binary_label = 1 if tb_type in ["possible", "confirmed", "probable"] and minority == 1 else 0
    return binary_label


# Proporciones de datos
# Aqui en un futuro vamos a usar 5-fold cross validation , no vamos a dividir los datos por fracciones
val_frac = 0.1
test_frac = 0.1
length = len(image_files_list)

# Mezclar índices aleatoriamente
indices = np.arange(length)
np.random.shuffle(indices)

# Dividir en test, validación y entrenamiento
test_split = int(test_frac * length)
val_split = int(val_frac * length) + test_split

test_indices = indices[:test_split]
val_indices = indices[test_split:val_split]
train_indices = indices[val_split:]

# Obtener solo etiquetas binarias
train_y_binary = [get_labels(image_files_list[i]) for i in train_indices]
val_y_binary = [get_labels(image_files_list[i]) for i in val_indices]
test_y_binary = [get_labels(image_files_list[i]) for i in test_indices]


# Imprimir tamaños correctos
print(f"Training count: {len(train_y_binary)}, Validation count: {len(val_y_binary)}, Test count: {len(test_y_binary)}")

#APLICAR TRANSFORMACIONES EN IMAGENES 

train_transforms = Compose(
    [
        LoadImage(image_only=True),  # Carga la imagen desde la ruta
        EnsureChannelFirst(),  # Asegura que tenga el formato (C, H, W)
        Resize((256,256)),
        EnsureType(),  # Convierte a Tensor si aún no lo es
        Lambda(lambda x: x if x.shape[0] == 3 else x.repeat(3, 1, 1)),  # Si la imagen no es de 1 canal, fuerza a 1 canal  
        ScaleIntensity(),  # Normaliza valores entre 0 y 1
        RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),  # Rotación aleatoria
        #RandFlip(spatial_axis=1, prob=0.5),  # Flip horizontal
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),  # Zoom aleatorio
    ]
)

val_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Resize((256, 256)),  # Asegurar tamaño uniforme
        EnsureType(),
        Lambda(lambda x: x if x.shape[0] == 3 else x.repeat(3, 1, 1)),  # Asegurar 3 canales
        ScaleIntensity(),  # Normalizar valores
    ]
)

y_pred_trans = Compose([Activations(softmax=True)])  # Softmax para predicciones
y_trans = Compose([AsDiscrete(to_onehot=2)])  # Convertir etiquetas a one-hot para 2 clases


#AJUSTAMOS LOS VALORES DEL DATASET 

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels_binary, transforms=None):
        self.image_files = image_files
        self.labels_binary = labels_binary
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = self.image_files[index]
        img = img_path

        if self.transforms:
            img = self.transforms(img)

        binary_label = torch.tensor(self.labels_binary[index], dtype=torch.float32)
        return img, binary_label


# 📌 Obtener rutas de imágenes separadas
train_image_files = [image_files_list[i] for i in train_indices]
val_image_files = [image_files_list[i] for i in val_indices]
test_image_files = [image_files_list[i] for i in test_indices]

# 📌 Dataset de entrenamiento, validación y test (solo binario)
train_ds = CustomDataset(train_image_files, train_y_binary, transforms=train_transforms)
val_ds = CustomDataset(val_image_files, val_y_binary, transforms=val_transforms)
test_ds = CustomDataset(test_image_files, test_y_binary, transforms=val_transforms)

# 📌 Dataloaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=32, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=32, num_workers=4)

# 🔍 Verificar formas del loader de validación
val_img, val_binary_label = next(iter(val_loader))
print("Imagen shape (validación):", val_img.shape)
print("Etiqueta binaria (validación):", val_binary_label)

# 🔍 Verificar formas del loader de entrenamiento
sample_img, sample_binary_label = next(iter(train_loader))
print("Imagen shape:", sample_img.shape)
print("Etiqueta binaria:", sample_binary_label)


# RED NEURONAL 
# Definimos la arquitectura del modelo
class BinaryClassifierDenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = DenseNet201(spatial_dims=2, in_channels=3, out_channels=512)
        self.fc_binary = nn.Linear(512, 1)

    def forward(self, x):
        x = self.backbone(x)
        binary_output = self.fc_binary(x)
        return binary_output

# ESTABLECEMOS EL MODELO Y LAS DISTINTAS MÉTRICAS
# Seleccionamos el dispositivo (GPU si está disponible, sino CPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Solo se verá la GPU con índice 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Usando el dispositivo: {device}")

# Inicializamos el modelo y lo movemos al dispositivo
model = BinaryClassifierDenseNet().to(device)

# Definimos las funciones de pérdida
loss_binary = torch.nn.BCEWithLogitsLoss()  # Para la clasificación binaria
loss_categorical = torch.nn.CrossEntropyLoss()  # Para la clasificación categórica

# Optimizador
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Parámetros de entrenamiento
max_epochs = 250
val_interval = 1  # Cada cuántas épocas evaluar
auc_metric = ROCAUCMetric()  # Métrica para la clasificación binaria

# Variables para almacenar métricas y pérdidas
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

# Para visualizar en TensorBoard
writer = SummaryWriter()
train_loss_values = []
val_loss_values = []
val_auc_values = []

for epoch in range(max_epochs):
    print(f"Epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0

    for batch_data in train_loader:
        images, binary_labels = batch_data
        images, binary_labels = images.to(device), binary_labels.to(device)

        optimizer.zero_grad()

        # 🔹 Forward pass
        binary_output = model(images)

        # 🔹 Calcular pérdida binaria
        loss_b = loss_binary(binary_output.squeeze(), binary_labels)

        # 🔹 Backward pass y optimización
        loss_b.backward()
        optimizer.step()

        epoch_loss += loss_b.item()

    # 🔹 Guardar pérdida promedio de la época
    epoch_loss /= len(train_loader)
    train_loss_values.append(epoch_loss)
    print(f" Pérdida de entrenamiento: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        val_loss = 0
        all_binary_preds = []
        all_binary_labels = []

        with torch.no_grad():
            for val_data in val_loader:
                images, binary_labels = val_data
                images, binary_labels = images.to(device), binary_labels.to(device)

                binary_output = model(images)

                # 🔹 Calcular pérdida binaria
                loss_b = loss_binary(binary_output.squeeze(), binary_labels)
                val_loss += loss_b.item()

                # 🔹 Guardar predicciones para evaluar AUC
                binary_probs = torch.sigmoid(binary_output).cpu().numpy()
                all_binary_preds.extend(binary_probs)
                all_binary_labels.extend(binary_labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_loss_values.append(val_loss)
        print(f" Pérdida de validación: {val_loss:.4f}")

        all_binary_preds = torch.tensor(np.array(all_binary_preds), dtype=torch.float32)
        all_binary_labels = torch.tensor(np.array(all_binary_labels), dtype=torch.float32)

        # Calcular AUC
        auc_metric(y_pred=all_binary_preds, y=all_binary_labels)
        auc_value = auc_metric.aggregate()
        auc_metric.reset()
        val_auc_values.append(auc_value)

        print(f" AUC en validación: {auc_value:.4f}")

        if auc_value > best_metric:
            best_metric = auc_value
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), "best_model.pth")
            print(" Modelo guardado con mejor AUC!")

print(f" Mejor AUC: {best_metric:.4f} en la época {best_metric_epoch}")
writer.close()
#EVALUAR EL MODELO 

# 🔹 Evaluación final del modelo en el conjunto de validación
model.eval()
all_binary_preds = []
all_binary_labels = []

with torch.no_grad():
    for val_data in val_loader:
        images, binary_labels = val_data  # Solo nos interesa la etiqueta binaria
        images = images.to(device)

        binary_output = model(images)  # Solo usamos la salida binaria
        binary_probs = torch.sigmoid(binary_output).cpu().numpy()
        binary_preds = (binary_probs > 0.7).astype(int)  # Convertimos probabilidades a etiquetas (0 o 1)

        all_binary_preds.extend(binary_preds)
        all_binary_labels.extend(binary_labels.cpu().numpy())

# 🔹 Calcular métricas
accuracy = accuracy_score(all_binary_labels, all_binary_preds)
precision = precision_score(all_binary_labels, all_binary_preds)
recall = recall_score(all_binary_labels, all_binary_preds)
f1 = f1_score(all_binary_labels, all_binary_preds)
conf_matrix = confusion_matrix(all_binary_labels, all_binary_preds)

# 🔹 Mostrar resultados
print(f"🔍 **Evaluación del Modelo** 🔍")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Matriz de Confusión:")
print(conf_matrix)


# 🎨 GRAFICAR CURVAS
epochs_range = list(range(1, len(train_loss_values) + 1))
val_epochs_range = list(range(val_interval, max_epochs + 1, val_interval))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_loss_values, label='Train Loss')
plt.plot(val_epochs_range, val_loss_values, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_epochs_range, val_auc_values, label='Validation AUC', color='green')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.title('Validation AUC Curve')
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
print("📈 Curvas de entrenamiento guardadas en 'training_curves.png'")














def predict_tuberculosis(model, image):
    """
    Usa la salida del modelo para determinar si hay tuberculosis.
    """
    model.eval()
    with torch.no_grad():
        # Obtener predicciones
        binary_pred, categorical_pred = model(image)

        # Convertir a probabilidades
        binary_prob = torch.sigmoid(binary_pred)  # [0, 1]
        categorical_prob = torch.softmax(categorical_pred, dim=1)  # [3 clases]

        # Decidir si es tuberculosis
        has_tuberculosis = (binary_prob > 0.5) and (categorical_prob.argmax() > 0)
        return has_tuberculosis