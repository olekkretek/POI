import numpy as np
import csv
from sklearn.cluster import DBSCAN
import pyransac3d as pyrsc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Wczytywanie pliku
def load_xyz(file_path):
    points = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            vals = [val for val in row if val]
            if len(vals) >= 3:
                try:
                    points.append([float(vals[0]), float(vals[1]), float(vals[2])])
                except ValueError:
                    continue
    return np.array(points)

#Funkcja DBSCAN 
def cluster_dbscan(points, eps=1.5, min_samples=10):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_
    clusters = [points[labels == i] for i in set(labels) if i != -1]
    noise = points[labels == -1]
    cluster_indices = {label: idx for idx, label in enumerate(sorted(set(labels) - {-1}))}
    return labels, clusters, noise, cluster_indices

# RANSAC
def fit_plane(points, threshold=0.2, max_iterations=1000):
    if len(points) < 3:
        return None
    plane = pyrsc.Plane()
    eq, inliers_idx = plane.fit(points, thresh=threshold, maxIteration=max_iterations)
    if inliers_idx is None or len(inliers_idx) == 0:
        return None
    normal = np.array(eq[:3])
    d = eq[3]
    normal /= np.linalg.norm(normal)
    return normal, d, points[inliers_idx], np.array(inliers_idx)

#Liczenie średniej odległości punktów od płaszczyzny
def average_distance(points, normal, d):
    num = np.abs(np.dot(points, normal) + d)
    denom = np.linalg.norm(normal)
    return np.mean(num / denom)

#Klasyfikacja płaszczyzny
def classify_plane(points, plane_result, inlier_threshold=0.8):
    if plane_result is None:
        print("  This cluster is not a plane.")
        return False, None

    normal, d, inliers, _ = plane_result
    inlier_ratio = len(inliers) / len(points)

    if inlier_ratio > 0.8:
        print("  This cluster is a plane.")
        verticality = abs(normal[2])
        if verticality > 0.9:
            print("  The plane is horizontal.")
        elif verticality < 0.1:
            print("  The plane is vertical.")
        else:
            print("  The plane is inclined.")
        return True, inlier_ratio
    else:
        print("  This cluster is not a plane.")
        return False, inlier_ratio

#Wizualizacja
def visualize(points, labels, plane_fit_results, cluster_indices, highlight_inliers=True):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10)[:3] for i in range(len(cluster_indices))]
    point_colors = np.full((points.shape[0], 3), 0.7)  # Szary dla szumu
    point_sizes = np.ones(points.shape[0]) * 5

    legend_handles = [] 
    for idx, label in enumerate(cluster_indices):
        mask = labels == label
        cluster_points = points[mask]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                    color=colors[idx], s=5, label=f"Klaster {idx}")
        point_colors[mask] = colors[idx]
    
    if highlight_inliers:
        for cl_idx, (normal, d, inliers, inliers_idx) in plane_fit_results.items():
            cluster_label = [k for k, v in cluster_indices.items() if v == cl_idx][0]
            cluster_mask = np.where(labels == cluster_label)[0]
            for idx_local in inliers_idx:
                if idx_local < len(cluster_mask):
                    point_sizes[cluster_mask[idx_local]] = 20

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=point_colors, s=point_sizes)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Klastrowanie DBSCAN i dopasowanie płaszczyzn RANSAC")
    ax.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    file_path = 'combined.xyz'

    points = load_xyz(file_path)
    if points.size == 0:
        exit("Brak danych!")

    labels, clusters, noise, cluster_indices = cluster_dbscan(points)

    plane_fit_results = {}
    for i, cluster in enumerate(clusters):
        result = fit_plane(cluster)
        print(f"\nKlaster {i}:")
        is_planar, inlier_ratio = classify_plane(cluster, result)
        
        if is_planar and result:
            normal, d, inliers, _ = result
            print(f"  Wektor normalny: {normal}")
            print(f"  Inlier ratio: {inlier_ratio:.2f}")
            plane_fit_results[i] = result
        else:
            if inlier_ratio is not None:
                print(f"  Inlier ratio: {inlier_ratio:.2f}")

    visualize(points, labels, plane_fit_results, cluster_indices)
