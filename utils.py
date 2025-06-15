from torch.utils.data import Dataset
from glob import glob
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors

class PCDataset(Dataset):
    def __init__(self, stage, transform=None):
        self.transform = transform
        self.stage = stage
        self.basedir = '/content/RGB2point'

        if stage == "train":
            image_paths = f"{self.basedir}/split/shapenet_train.txt"
        elif stage == "test":
            image_paths = f"{self.basedir}/split/shapenet_test.txt"

        with open(image_paths) as caption_file:
            self.filenames = caption_file.readlines()

        self.numbers_list = [f"{i:02}" for i in range(24)]

        labels = []
        category = set()
        for f in self.filenames:
            attr = f.split("/")
            labels.append(attr[1].strip())
            category.add(attr[0])

        category = list(category)
        self.labels = []
        self.data = []

        for c in ["02958343", "02691156", "03001627"]:
            for label in labels:
                volume_path = f"{self.basedir}/{c}/{label}/pointcloud_1024.npy"
                files = glob(
                    f"{self.basedir}/{c}/{label}/rendering/*.png"
                )
                for file in files:
                    if self.stage == "train":
                        if os.path.exists(volume_path):
                            self.data.append([c, label, file])

                if self.stage == "test":
                    if os.path.exists(volume_path) and len(files) > 1:
                        test_image_path = f"{self.basedir}/{c}/{label}/rendering/00.png"
                        self.data.append([c, label, test_image_path])

    def __len__(self):
        return len(self.data)

    def normalize_point_cloud(self, point_cloud):
        """
        Normalize a point cloud to be centered around the origin and fit within a unit cube.

        :param point_cloud: Numpy array of shape (num_points, dimensions)
        :return: Normalized point cloud.
        """
        centroid = np.mean(point_cloud, axis=0)
        centered_point_cloud = point_cloud - centroid
        if self.stage == "train":
            np.random.shuffle(centered_point_cloud)
        return centered_point_cloud

    def __getitem__(self, idx):
        data = self.data[idx]
        category = data[0]
        label = data[1]
        image = data[2]

        image_files = [image]
        pc = np.load(
            f"{self.basedir}/{category}/{label}/pointcloud_1024.npy"
        )
        pc = self.normalize_point_cloud(pc)

        images = []
        for filename in image_files:
            image = Image.open(filename).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)

        name = f"{category}_{label}"
        images_tensor = torch.stack(images, dim=0)

        return images_tensor, torch.as_tensor(pc, dtype=torch.float32), name
    


def chamfer_distance(x, y, metric="l2", direction="bi"):
    """Chamfer distance between two point clouds

    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """

    if direction == "y_to_x":
        x_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == "x_to_y":
        y_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == "bi":
        x_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: 'y_x', 'x_y', 'bi'")

    return chamfer_dist

def fscore(dist1, dist2, threshold=0.01):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances, so you should adapt the threshold accordingly.
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2
    
class EMDLoss(nn.Module):
    def __init__(self):
        super(EMDLoss, self).__init__()

    def forward(self, pred, target):
        # pred and target are expected to have shape (batch_size, 1024, 3)
        assert pred.shape == target.shape
        assert pred.shape[1] == 1024 and pred.shape[2] == 3

        batch_size = pred.shape[0]
        num_points = pred.shape[1]

        # Compute pairwise distances between all points
        diff = pred.unsqueeze(2) - target.unsqueeze(1)
        dist = torch.sum(diff**2, dim=-1)

        # Solve the assignment problem using Hungarian algorithm
        # Note: This is a simplified version and may not be the most efficient for large point clouds
        assignment = torch.zeros_like(dist)
        for b in range(batch_size):
            _, indices = torch.topk(dist[b], k=num_points, largest=False, dim=1)
            assignment[b] = torch.scatter(assignment[b], 1, indices, 1)

        # Compute the EMD
        emd = torch.sum(dist * assignment, dim=[1, 2]) / num_points

        return emd.mean()


import open3d as o3d
def export_to_ply(point_cloud, filename):
    """
    Export a point cloud to a PLY file.
    :param point_cloud: Numpy array of shape (num_points, 3) representing the point cloud.
    :param filename: String, the name of the file to save the point cloud to.
    """
    # Convert numpy array to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Write to a PLY file
    o3d.io.write_point_cloud(filename, pcd)


from torchvision import transforms
from PIL import Image
def predict(model, image_path, save_path):

    # Define the transformations
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Apply the transformations
    input_tensor = transform(image)
    input_tensor = input_tensor.reshape(1,1,3,224,224)

    

    # Invoke the model
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(input_tensor)


    export_to_ply(output[0], save_path)
    print(f"Image from {image_path} saved to {save_path}")

import matplotlib.pyplot as plt
import argparse

def renderPointCloud(fileName, renderingLib = 'o3d'):
    if fileName.endswith('.npy'):
        try:
            points = np.load(fileName)
            if points.shape[1] != 3:
                print("Error: The point cloud data must have 3 columns (x, y, z).")
                return
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)
        except FileNotFoundError:
            print(f"Error: {fileName} not found. Please provide the correct path.")
            return
    elif fileName.endswith('.ply'):
        try:
            point_cloud = o3d.io.read_point_cloud(fileName)
            points = np.asarray(point_cloud.points)
            if points.shape[0] == 0:
                print(f"Error: {fileName} not found. Please provide the correct path.")
                return
        except:
            print(f"Error: {fileName} not found. Please provide the correct path.")
            return

    # Check if the data has the correct shape
    print(f'Shape of point cloud: {points.shape}')
    #unique_vals = np.unique(points)
    #num_uniques = len(unique_vals)
    #print(f'unique values in point cloud: {unique_vals}, number of values {num_uniques}')

    if renderingLib == 'o3d':
        o3d.visualization.draw_geometries([point_cloud])
    elif renderingLib == 'plt':
        # Separate the x, y, and z coordinates
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]


        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot the points
        ax.scatter(x, y, z, s=2)  # s is the size of the points

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Show the plot
        plt.title('Point Cloud Visualization')
        plt.show()

if __name__ == '__main__':
    # Load the .npy file
    fileName = 'ShapeNet_pointclouds/04379243/f84b49a7d9e6f31ae0c8359faeb78bbe/pointcloud_2048.npy'
    '''
    fileName = 'ShapeNet_pointclouds/04379243/f80427c47d2fc4169ccce4c6d5bb195f/pointcloud_2048.npy'
    fileName = 'ShapeNet_pointclouds/04379243/f86ad1699aa2da6dc955e5ed03ef3a2f/pointcloud_2048.npy'
    fileName = 'ShapeNet_pointclouds/04379243/f99ebf0f053140525a0e5699b3040a35/pointcloud_2048.npy'
    fileName = 'ShapeNet_pointclouds/02691156/fe23572aac36c84761cfaa00f7177470/pointcloud_2048.npy'
    fileName = 'ShapeNet_pointclouds/02691156/dd4da4309c12d47bc2c2c81e2232aa95/pointcloud_2048.npy'
    fileName = 'ShapeNet_pointclouds/02691156/676e568ab8e286ad67c54439d6177032/pointcloud_2048.npy'
    fileName = 'ShapeNet_pointclouds/02691156/10eeb119fd5508e0d6d949577c389a84/pointcloud_2048.npy'
    fileName = 'ShapeNet_pointclouds/02958343/fcbf9e1dd55ab172ce27281f3b76d1f5/pointcloud_2048.npy'
    fileName = 'ShapeNet_pointclouds/02958343/e5fea12dcf4bb5b250c1ed9db9037190/pointcloud_2048.npy'
    fileName = 'ShapeNet_pointclouds/02958343/84e4dd1c6306a582a97e62e669e9544d/pointcloud_2048.npy'
    fileName = 'ShapeNet_pointclouds/02958343/4a5bd025a2500b3b3967c36b4916b4de/pointcloud_2048.npy'
    fileName = 'ShapeNet_pointclouds/02958343/fc28356cd7948b90466f54ef02e60f2f/pointcloud_2048.npy'
    fileName = 'ShapeNet_pointclouds/02958343/fc28356cd7948b90466f54ef02e60f2f/pointcloud_1024.npy'
    fileName = 'ShapeNet_pointclouds/02958343/707dd25fa8ebe7b73a6cc070f17a46d7/pointcloud_2048.npy'
    fileName = 'ShapeNet_pointclouds/02958343/437f3ed08c32f2b9092ea6b09901598/pointcloud_2048.npy'
    fileName = 'ShapeNet_pointclouds/02958343/163a2c8ead6872ad7ae33d942430658c/pointcloud_2048.npy'
    fileName = 'ShapeNet_pointclouds/02958343/2d6c360e7d8ff4162e603610d34efc63/pointcloud_2048.npy'
    fileName = 'ShapeNet_pointclouds/02958343/38afcfa425487df5fff77b62a794a2d8/pointcloud_2048.npy'
    '''
    fileName = '1013.ply'

    parser = argparse.ArgumentParser("vis_point_cloud")
    parser.add_argument('--file', default=fileName, type=str, help='Point Cloud file name')
    parser.add_argument('--lib', default='o3d', type=str, help='o3d/plt')

    args = parser.parse_args()
    renderPointCloud(args.file, args.lib)
