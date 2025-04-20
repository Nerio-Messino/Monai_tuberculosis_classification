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
    RandGaussianNoise,
    RandAdjustContrast,
    RandShiftIntensity


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

#APLICAR TRANSFORMACIONES EN IMAGENES 
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

train_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Resize((256, 256)),
        EnsureType(),
        Lambda(lambda x: x if x.shape[0] == 3 else x.repeat(3, 1, 1)),
        ScaleIntensity(),

        # ⬇️ Aumentos anatómicamente consistentes
        RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),  # Ya estaba, perfecto
        RandFlip(spatial_axis=1, prob=0.5),  # Flip vertical (eje 1) - bien para rayos
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),

        RandGaussianNoise(prob=0.2),  # Añade un poco de ruido
        RandAdjustContrast(prob=0.3, gamma=(0.9, 1.1)),  # Ligeras variaciones de contraste
        RandShiftIntensity(offsets=0.1, prob=0.3),  # Cambios leves en intensidad
    ]
)

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

# 🔢 Selecciona el índice de la GPU del sistema que quieres usar (0 o 1, según nvitop/nvidia-smi)
gpu_id = 0  # ← CAMBIA ESTE NÚMERO para usar la GPU 0 o 1 del sistema

# 🧠 Restringe la visibilidad solo a esa GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

# 🖥️ PyTorch solo verá 1 GPU, que será 'cuda:0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Usando el dispositivo: {device} (GPU real: {gpu_id})")

# Proporciones de datos
df = df[df["fold_cv"].notna()]
df["fold_cv"] = df["fold_cv"].astype(int)
folds = df["fold_cv"].unique()
folds.sort()
all_fold_metrics = []
fold_results = []
num_folds = 5
folds_data = []

for fold in folds:
    # Test: fold actual
    test_indices = sorted(df[df["fold_cv"] == fold].index.tolist())  # Ordena los índices
    # Validación: siguiente fold (en ciclo)
    val_fold = (fold + 1) % num_folds
    val_indices = sorted(df[df["fold_cv"] == val_fold].index.tolist())  # Ordena los índices
    # Entrenamiento: el resto
    train_indices = sorted(df[~df["fold_cv"].isin([fold, val_fold])].index.tolist())  # Ordena los índices
    
    folds_data.append((train_indices, val_indices, test_indices))

print(f"Longitud de image_files_list: {len(image_files_list)}")  # Verifica que tenga 877 elementos

# ✅ Al principio, nómbrala diferente
binary_labels_full = np.array([get_labels(img_path) for img_path in image_files_list])


# 🔁 Bucle de entrenamiento por fold
for fold, (train_indices, val_indices, test_indices) in enumerate(folds_data):
    print(f"\n🚀 Fold {fold + 1}/{num_folds}")
    
    # Preparar datos específicos del fold
    train_image_files = [image_files_list[i] for i in train_indices]
    val_image_files = [image_files_list[i] for i in val_indices]
    test_image_files = [image_files_list[i] for i in test_indices]

    binary_labels = np.array([get_labels(img_path) for img_path in image_files_list])
    
    train_y_binary = binary_labels[train_indices]
    val_y_binary = binary_labels[val_indices]
    test_y_binary = binary_labels[test_indices]

    # Datasets y dataloaders
    train_ds = CustomDataset(train_image_files, train_y_binary, transforms=train_transforms)
    val_ds = CustomDataset(val_image_files, val_y_binary, transforms=val_transforms)
    test_ds = CustomDataset(test_image_files, test_y_binary, transforms=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=32, num_workers=4)

    # 🧠 Modelo, pérdida, optimizador
    model = BinaryClassifierDenseNet().to(device)
    loss_binary = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    auc_metric = ROCAUCMetric()

    # 🎯 Parámetros
    max_epochs = 200
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1

    # 📊 Para curvas
    train_loss_values = []
    val_loss_values = []
    val_auc_values = []

    writer = SummaryWriter()

    # 🔁 Entrenamiento por epoch
    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0

        for batch_data in train_loader:
            images, binary_labels = batch_data
            images, binary_labels = images.to(device), binary_labels.to(device)

            optimizer.zero_grad()
            binary_output = model(images)
            loss_b = loss_binary(binary_output.squeeze(), binary_labels)
            loss_b.backward()
            optimizer.step()
            epoch_loss += loss_b.item()

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
                    loss_b = loss_binary(binary_output.squeeze(), binary_labels)
                    val_loss += loss_b.item()

                    binary_probs = torch.sigmoid(binary_output).cpu().numpy()
                    all_binary_preds.extend(binary_probs)
                    all_binary_labels.extend(binary_labels.cpu().numpy())

            val_loss /= len(val_loader)
            val_loss_values.append(val_loss)
            print(f" Pérdida de validación: {val_loss:.4f}")

            all_binary_preds = torch.tensor(np.array(all_binary_preds), dtype=torch.float32)
            all_binary_labels = torch.tensor(np.array(all_binary_labels), dtype=torch.float32)

            auc_metric(y_pred=all_binary_preds, y=all_binary_labels)
            auc_value = auc_metric.aggregate()
            auc_metric.reset()
            val_auc_values.append(auc_value)

            print(f" AUC en validación: {auc_value:.4f}")

            if auc_value > best_metric:
                best_metric = auc_value
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), f"best_model_fold{fold}.pth")
                print(" Modelo guardado con mejor AUC!")

    print(f"🏁 Fold {fold + 1} finalizado - Mejor AUC: {best_metric:.4f} (época {best_metric_epoch})")
    writer.close()

    # 📈 Guardar curva de entrenamiento
    epochs_range = list(range(1, len(train_loss_values) + 1))
    val_epochs_range = list(range(val_interval, max_epochs + 1, val_interval))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss_values, label='Train Loss')
    plt.plot(val_epochs_range, val_loss_values, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve - Fold {fold}')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_epochs_range, val_auc_values, label='Validation AUC', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.title(f'Validation AUC Curve - Fold {fold}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"training_curves_fold{fold}.png")
    print(f"📊 Curvas guardadas como training_curves_fold{fold}.png")

    # ✅ Evaluación final del mejor modelo
    model.load_state_dict(torch.load(f"best_model_fold{fold}.pth"))
    model.eval()
    all_binary_preds = []
    all_binary_labels = []

    with torch.no_grad():
        for val_data in val_loader:
            images, binary_labels = val_data
            images = images.to(device)

            binary_output = model(images)
            binary_probs = torch.sigmoid(binary_output).cpu().numpy()
            binary_preds = (binary_probs > 0.7).astype(int)

            all_binary_preds.extend(binary_preds)
            all_binary_labels.extend(binary_labels.cpu().numpy())

    accuracy = accuracy_score(all_binary_labels, all_binary_preds)
    precision = precision_score(all_binary_labels, all_binary_preds)
    recall = recall_score(all_binary_labels, all_binary_preds)
    f1 = f1_score(all_binary_labels, all_binary_preds)
    conf_matrix = confusion_matrix(all_binary_labels, all_binary_preds)

    print(f"📍 Resultados evaluación final - Fold {fold}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Matriz de Confusión:")
    print(conf_matrix)

    fold_results.append({
        "fold": int(fold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "best_auc": best_metric,
    })


# 🧾 Resumen general tras todos los folds
print("\n📋 Resultados por fold:")
for res in fold_results:
    print(f"Fold {res['fold']}: Acc={res['accuracy']:.4f}, Prec={res['precision']:.4f}, Rec={res['recall']:.4f}, F1={res['f1']:.4f}, AUC={res['best_auc']:.4f}")

# Promedio general
avg_metrics = {
    "accuracy": np.mean([res["accuracy"] for res in fold_results]),
    "precision": np.mean([res["precision"] for res in fold_results]),
    "recall": np.mean([res["recall"] for res in fold_results]),
    "f1": np.mean([res["f1"] for res in fold_results]),
    "auc": np.mean([res["best_auc"] for res in fold_results])
}

print("\n📈 Promedio final en todos los folds:")
for metric, value in avg_metrics.items():
    print(f"{metric.capitalize()}: {value:.4f}")

# 📊 Convertir resultados por fold en DataFrame
results_df = pd.DataFrame(fold_results)

# 📌 Redondear todo a 4 decimales
results_df = results_df.round(4)

# ✅ Calcular media y desviación estándar
mean_row = results_df.mean().round(4)
std_row = results_df.std().round(4)

# 🧮 Crear fila final con "media ± std"
summary_row = [f"{mean:.4f} ± {std:.4f}" for mean, std in zip(mean_row, std_row)]

# Añadir fila con etiqueta y combinar todo
summary_df = pd.DataFrame([summary_row], columns=results_df.columns, index=["Mean ± Std"])
final_df = pd.concat([results_df, summary_df])

# 💾 Guardar a CSV (opcional)
final_df.to_csv("cv_metrics_summary.csv", index=False)

# 📋 Mostrar la tabla bonita
print("\n📊 Tabla de métricas por fold con media ± desviación estándar:\n")
print(final_df.to_markdown())

print("\n🧠 Iniciando inferencia con ensemble en test...")

# 1️⃣ Obtener todos los modelos guardados
ensemble_models = []
for fold in range(num_folds):
    model = BinaryClassifierDenseNet().to(device)
    model.load_state_dict(torch.load(f"best_model_fold{fold}.pth"))
    model.eval()
    ensemble_models.append(model)

# 2️⃣ Preparar conjunto de test completo
# Aquí recogemos todos los índices que han sido usados como test en cada fold
all_test_indices = [idx for _, _, test_idx in folds_data for idx in test_idx]
unique_test_indices = sorted(list(set(all_test_indices)))

test_image_files = [image_files_list[i] for i in unique_test_indices]
test_y_binary = binary_labels_full[unique_test_indices]

test_ds = CustomDataset(test_image_files, test_y_binary, transforms=val_transforms)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

# 3️⃣ Inferencia con ensemble (soft voting)
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.cpu().numpy()
        
        # Predicciones de todos los modelos
        fold_probs = []
        for model in ensemble_models:
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            fold_probs.append(probs)
        
        # Media de probabilidades (soft voting)
        mean_probs = np.mean(fold_probs, axis=0)
        final_preds = (mean_probs > 0.7).astype(int)  # Umbral
        
        all_preds.extend(final_preds)
        all_labels.extend(labels)

# 4️⃣ Métricas finales
ensemble_accuracy = accuracy_score(all_labels, all_preds)
ensemble_precision = precision_score(all_labels, all_preds)
ensemble_recall = recall_score(all_labels, all_preds)
ensemble_f1 = f1_score(all_labels, all_preds)
conf_matrix = confusion_matrix(all_labels, all_preds)

print("\n✅ Resultados del Ensemble en Test:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Matriz de Confusión:")
print(conf_matrix)

# 📌 Guardar resultados del ensemble en test
ensemble_results = {
    "fold": "Ensemble-Test",
    "accuracy": ensemble_accuracy,
    "precision": ensemble_precision,
    "recall": ensemble_recall,
    "f1": ensemble_f1,
}

# 📌 Añadirlo al DataFrame final
final_df_with_ensemble = pd.concat([results_df, pd.DataFrame([ensemble_results]), summary_df])

# 💾 Guardar todo a un nuevo CSV
final_df_with_ensemble.to_csv("final_results_with_ensemble.csv", index=False)

# 📋 Mostrar la tabla completa
print("\n🧾 Tabla completa (CV + Ensemble Test):\n")
print(final_df_with_ensemble.to_markdown())
