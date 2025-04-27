import numpy as np

def generate_horizontal_plane(width, length, num_points):
    x = np.random.uniform(-width / 2, width / 2, num_points)
    y = np.random.uniform((-length / 2)+25,(length / 2)+25, num_points)
    z = np.zeros(num_points)
    return np.column_stack((x, y, z))

def generate_vertical_plane(width, height, num_points):
    x = np.zeros(num_points)
    y = np.random.uniform(-width / 2, width / 2, num_points)
    z = np.random.uniform(-height / 2, height / 2, num_points)
    return np.column_stack((x, y, z))

def generate_cylinder(radius, height, num_points):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    z = np.random.uniform(-height / 2, height / 2, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)+12
    return np.column_stack((x, y, z))

def save_to_xyz(filename, points):
    np.savetxt(filename, points, fmt='%.6f', delimiter=' ')

def merge_xyz_files(output_filename, input_filenames):
    merged_data = []
    for filename in input_filenames:
        try:
            data = np.loadtxt(filename, delimiter=' ')
            merged_data.append(data)
        except Exception as e:
            print(f"Błąd podczas wczytywania {filename}: {e}")
    if merged_data:
        all_points = np.vstack(merged_data)
        np.savetxt(output_filename, all_points, fmt='%.6f', delimiter=' ')

if __name__ == "__main__":
    num_points = 1000

    horizontal_plane = generate_horizontal_plane(10, 10, num_points)
    save_to_xyz("horizontal_plane.xyz", horizontal_plane)

    vertical_plane = generate_vertical_plane(10, 10, num_points)
    save_to_xyz("vertical_plane.xyz", vertical_plane)

    cylinder = generate_cylinder(5, 10, num_points)
    save_to_xyz("cylinder.xyz", cylinder)

    # Scalanie danych z trzech plików do jednego
    merge_xyz_files("combined.xyz", ["horizontal_plane.xyz", "vertical_plane.xyz", "cylinder.xyz"])

    print("Wszystkie pliki zostały scalone w combined.xyz.")

