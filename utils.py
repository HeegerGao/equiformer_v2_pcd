import torch
from copy import deepcopy

def translate_point_cloud(point_cloud, translation_vector):
    return point_cloud + translation_vector

def rotate_point_cloud(point_cloud, rotation_matrix):
    return torch.matmul(point_cloud, rotation_matrix.T)

def generate_random_rotation_matrix(device="cuda:7"):
    theta = torch.rand(1) * 2 * torch.pi
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
    rotation_matrix = torch.tensor([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    return rotation_matrix.to(device)

def test_equivariance(model, data, device="cuda:7"):
    model.eval()
    with torch.no_grad():
        data_copy = deepcopy(data)

        heatmap, orientation = model(data_copy)
        heatmap, orientation = heatmap.detach().cpu(), orientation.detach().cpu()

        translation_vector = torch.tensor([1.0, 0.0, -1.0]).to(device)
        data_copy.pos = translate_point_cloud(data_copy.pos, translation_vector)
        translated_heatmap, trasnlated_orientation = model(data_copy)
        translated_heatmap, trasnlated_orientation = translated_heatmap.detach().cpu(), trasnlated_orientation.detach().cpu()

        data_copy = deepcopy(data)
        rotation_matrix = generate_random_rotation_matrix(device)
        data_copy.pos = rotate_point_cloud(data_copy.pos, rotation_matrix)
        data_copy.feature[:, 3:] = rotate_point_cloud(data_copy.feature[:, 3:], rotation_matrix)
        rotated_heatmap, rotated_orientation = model(data_copy)
        rotated_heatmap, rotated_orientation = rotated_heatmap.detach().cpu(), rotated_orientation.detach().cpu()

        translation_invariance_heatmap = torch.allclose(heatmap, translated_heatmap, atol=1e-5)
        translation_invariance_orientation = torch.allclose(orientation, trasnlated_orientation, atol=1e-5)
        print("Translation Invariance (Heatmap):", translation_invariance_heatmap)
        print("Translation Invariance (Orientation):", translation_invariance_orientation)

        rotation_invariance = torch.allclose(heatmap, rotated_heatmap, atol=1e-5)
        rotation_equivariance = torch.allclose(rotate_point_cloud(orientation.reshape(-1, 3), rotation_matrix.detach().cpu()), rotated_orientation.reshape(-1, 3), atol=1e-5)
        print("Rotation Invariance:", rotation_invariance)
        print("Rotation Equivariance:", rotation_equivariance)