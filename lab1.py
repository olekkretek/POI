import numpy as np

def generate_horizontal_plane(width, length, num_points):
    x = np.random.uniform(0, width, num_points)
    y = np.random.uniform(0, length, num_points)
    z = np.zeros(num_points)
    return np.column_stack((x, y, z))

def generate_vertical_plane(width, height, num_points):
    x = np.random.uniform(0, width, num_points)
    y = np.zeros(num_points)
    z = np.random.uniform(0, height, num_points)
    return np.column_stack((x, y, z))

def generate_cylinder(radius, height, num_points):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    z = np.random.uniform(0, height, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack((x, y, z))

def save_to_xyz(filename, points):
    np.savetxt(filename, points, fmt='%.6f', delimiter=' ')

if __name__ == "__main__":
    num_points = 1000
    
    horizontal_plane = generate_horizontal_plane(10, 10, num_points)
    save_to_xyz("horizontal_plane.xyz", horizontal_plane)
    
    vertical_plane = generate_vertical_plane(10, 10, num_points)
    save_to_xyz("vertical_plane.xyz", vertical_plane)
    
    cylinder = generate_cylinder(5, 10, num_points)
    save_to_xyz("cylinder.xyz", cylinder)
    
    print("Pliki xyz zostały zapisane.")

