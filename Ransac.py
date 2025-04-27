import csv 
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D 

file_path = 'combined.xyz'
#Wczytywanie pliku
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

def fit_plane_ransac(points, threshold=0.1, iterations=1000):
    best_eq = None
    best_inliers = 0
    best_normal = None

    for _ in range(iterations):
        sample = points[np.random.choice(points.shape[0], 3, replace=False)]
        p1, p2, p3 = sample

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        if np.linalg.norm(normal) == 0:
            continue

        normal = normal / np.linalg.norm(normal)
        d = -np.dot(normal, p1)

        distances = np.abs(np.dot(points, normal) + d)
        inliers = np.sum(distances < threshold)

        if inliers > best_inliers:
            best_inliers = inliers
            best_eq = (normal, d)
            best_normal = normal

    if best_eq is None:
        return np.array([0, 0, 0]), float('inf'), 0.0

    distances = np.abs(np.dot(points, best_normal) + best_eq[1])
    mean_distance = np.mean(distances)
    inlier_ratio = best_inliers / points.shape[0]
    return best_normal, mean_distance, inlier_ratio


point_cloud_data = load_xyz(file_path)

if point_cloud_data.shape[0] == 0:
    print("Exiting: No data points to cluster.")
else:
    print(f"Loaded {point_cloud_data.shape[0]} points.")

    k = 3
    if point_cloud_data.shape[0] < k:
        print(f"Warning: Number of points < k. Adjusting k to {point_cloud_data.shape[0]}.")
        k = point_cloud_data.shape[0]

    if k > 0:
        print(f"Applying K-means with k={k}...")
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', max_iter=300, random_state=42)
        labels = kmeans.fit_predict(point_cloud_data)
        print("K-means fitting complete.")
        print(f"Cluster centers found at:\n{kmeans.cluster_centers_}")

        disjoint_clouds = []
        for i in range(k):
            cluster_points = point_cloud_data[labels == i]
            disjoint_clouds.append(cluster_points)
            print(f"Cluster {i}: Found {cluster_points.shape[0]} points.")


        # Dopasowywanie płaszczyzn i klasyfikacja
        print("\n--- Wyniki dopasowania płaszczyzn ---")
        for i, cloud in enumerate(disjoint_clouds):
            if cloud.shape[0] < 3:
                print(f"Cluster {i}: too few points to fit a plane.")
                continue

            normal_vector, mean_dist, inlier_ratio = fit_plane_ransac(cloud, threshold=0.1)

            print(f"\nCluster {i}:")
            print(f"  Wektor normalny: {normal_vector}")
            print(f"  Średnia odległość do płaszczyzny: {mean_dist:.5f}")
            print(f"  Inlier ratio: {inlier_ratio:.2f}")

            if inlier_ratio > 0.8:
                print("  This cluster is a plane.")
                verticality = abs(normal_vector[2])
                if verticality > 0.9:
                    print("  The plane is horizontal.")
                elif verticality < 0.1:
                    print("  The plane is vertical.")
                else:
                    print("  The plane is inclined.")
            else:
                print("  This cluster is not a plane.")
# Wizualizacja
        print("\nAttempting visualization (optional)...")
        try:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            colors = plt.cm.viridis(np.linspace(0, 1, k))
            for i in range(k):
                cloud = disjoint_clouds[i]
                if cloud.shape[0] > 0:
                    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2],
                               label=f'Cluster {i}', c=[colors[i]], s=5)
            centers = kmeans.cluster_centers_
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                       marker='X', s=200, c='red', label='Centroids')
            ax.set_title('Point Cloud Clustered with K-means (k=3)')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            plt.show()
        except Exception as e:
            print(f"Plotting error: {e}")