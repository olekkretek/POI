import os
import numpy as np
import cv2 
from skimage import io, color
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import glob

# Konfiguracja
INPUT_IMAGE_DIR = r'C:\Users\olekk\OneDrive\Pulpit\drugistopien\POI\kody\lab3\lab_3_zdjecia'  #Obrazy oryginalne
PATCHES_OUTPUT_DIR = '\wycinki'   # Katalog, gdzie zapisane zostaną wycięte próbki
FEATURES_CSV_FILE = 'features.csv' # Plik CSV do zapisu cech
PATCH_SIZE = (128, 128)                 # Rozmiar wycinanych próbek (wysokość, szerokość)
GLCM_DISTANCES = [1, 3, 5]              # Odległości dla GLCM
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4] # Kąty dla GLCM (0, 45, 90, 135 stopni)
GLCM_LEVELS = 64                        # Liczba poziomów szarości

# Wczytywanie obrazów i wycinanie próbek
def extract_patches(input_dir, output_dir, patch_h, patch_w):
    os.makedirs(output_dir, exist_ok=True)
    categories = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    total_patches_saved = 0
    for category in categories:
        category_input_path = os.path.join(input_dir, category)
        category_output_path = os.path.join(output_dir, category)
        os.makedirs(category_output_path, exist_ok=True)
        image_files = glob.glob(os.path.join(category_input_path, '*.png')) + \
                      glob.glob(os.path.join(category_input_path, '*.jpg')) + \
                      glob.glob(os.path.join(category_input_path, '*.jpeg')) + \
                      glob.glob(os.path.join(category_input_path, '*.bmp')) + \
                      glob.glob(os.path.join(category_input_path, '*.tif'))

        category_patches_count = 0
        for img_path in image_files:
            try:
                img = cv2.imread(img_path)

                img_h, img_w = img.shape[:2]
                img_filename = os.path.basename(img_path)
                img_name_part = os.path.splitext(img_filename)[0]

                for r in range(0, img_h - patch_h + 1, patch_h):
                    for c in range(0, img_w - patch_w + 1, patch_w):
                        patch = img[r:r + patch_h, c:c + patch_w]

                        # Zapisywanie próbki
                        patch_filename = f"{img_name_part}_patch_{r:04d}_{c:04d}.png"
                        patch_output_path = os.path.join(category_output_path, patch_filename)
                        cv2.imwrite(patch_output_path, patch)
                        category_patches_count += 1

            except Exception as e:
                print(f"BŁĄD podczas przetwarzania obrazu {img_path}: {e}")

        print(f"Zapisano {category_patches_count} próbek dla kategorii '{category}'.")
        total_patches_saved += category_patches_count

# Wczytywanie próbek i wyznaczanie cech GLCM
def calculate_glcm_features(patches_dir, distances, angles, levels):

    categories = [d for d in os.listdir(patches_dir) if os.path.isdir(os.path.join(patches_dir, d))]
    
    all_features = []
    properties = ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM']

    for category in categories:
        print(f"Przetwarzanie kategorii: {category}")
        category_path = os.path.join(patches_dir, category)
        patch_files = glob.glob(os.path.join(category_path, '*.png'))

        processed_patches = 0
        for patch_path in patch_files:
            try:
                patch = io.imread(patch_path)
                # Konwersja do skali szarości
                if patch.ndim == 3:
                    patch_gray = color.rgb2gray(patch)
                elif patch.ndim == 2:
                    patch_gray = patch
                else:
                    continue
                if patch_gray.max() <= 1.0:
                     patch_gray = (patch_gray * 255).astype(np.uint8)
                else:
                     patch_gray = patch_gray.astype(np.uint8)
                img_quantized = np.floor(patch_gray / 256. * levels).astype(np.uint8)

                # Obliczenie macierzy GLCM
                glcm = graycomatrix(img_quantized,
                                    distances=distances,
                                    angles=angles,
                                    levels=levels,
                                    symmetric=True,
                                    normed=True)

                # Obliczenie cech GLCM
                feature_vector = {'category': category, 'patch_file': os.path.basename(patch_path)}
                for prop in properties:
                    prop_values = graycoprops(glcm, prop)
                    for i, dist in enumerate(distances):
                        feature_vector[f'{prop}_d{dist}'] = np.mean(prop_values[i, :])
                all_features.append(feature_vector)
                processed_patches += 1

            except Exception as e:
                print(f"BŁĄD podczas przetwarzania próbki {patch_path}: {e}")

    return all_features

# Zapis wektorów cech do pliku CSV
def save_features_to_csv(features_list, csv_filepath):
    try:
        df = pd.DataFrame(features_list)
        if 'category' in df.columns:
             cols = ['category'] + [col for col in df.columns if col != 'category']
             df = df[cols]
        df.to_csv(csv_filepath, index=False)
        print(f"Zapisano {len(df)} wektorów cech do pliku: {csv_filepath}")
    except Exception as e:
        print(f"BŁĄD podczas zapisywania pliku CSV {csv_filepath}: {e}")

# Klasyfikacja wektorów cech
def classify_features(csv_filepath, test_size=0, random_state=0, classifier_type=0):

    try:
        # Wczytywanie danych
        df = pd.read_csv(csv_filepath)
        print(f"Wczytano {len(df)} wektorów cech z {csv_filepath}.")

        if df.empty:
            print("BŁĄD: Plik CSV jest pusty.")
            return
        if 'category' not in df.columns:
            print("BŁĄD: Brak kolumny 'category' w pliku CSV.")
            return

        # Przygotowanie danych
        X = df.drop(['category', 'patch_file'], axis=1) # Cechy
        y_raw = df['category']                     # Etykiety

        # Kodowanie etykiet (zamiana nazw kategorii na liczby)
        le = LabelEncoder()
        y = le.fit_transform(y_raw)

        # Podział na zbiór treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y 
        )
        print(f"Podzielono dane na zbiór treningowy ({len(X_train)} próbek) i testowy ({len(X_test)} próbek).")

        # Wybór i trenowanie klasyfikatora
        if classifier_type.lower() == 'knn':
            print("(KNN):")
            model = KNeighborsClassifier(n_neighbors=5)
        elif classifier_type.lower() == 'svm':
            print("(SVM):")
            model = SVC(kernel='linear',  random_state=random_state)
 
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Dokładność (Accuracy) na zbiorze testowym: {accuracy:.4f} ({accuracy*100:.2f}%)")

    except Exception as e:
        print(f"BŁĄD podczas klasyfikacji: {e}")

if __name__ == "__main__":
    extract_patches(INPUT_IMAGE_DIR, PATCHES_OUTPUT_DIR, PATCH_SIZE[0], PATCH_SIZE[1])
    features = calculate_glcm_features(PATCHES_OUTPUT_DIR, GLCM_DISTANCES, GLCM_ANGLES, GLCM_LEVELS)
    save_features_to_csv(features, FEATURES_CSV_FILE)
    classify_features(FEATURES_CSV_FILE, test_size=0.90, random_state=42, classifier_type='svm')
    classify_features(FEATURES_CSV_FILE, test_size=0.2, random_state=42, classifier_type='knn')