import os 
import trimesh
import torch
import json
import pyfqmr
import numpy as np
import polyscope as ps
from pathlib import Path
from models.dilated_tooth_seg_network import LitDilatedToothSegmentationNetwork
import random
from teeth_numbering import color_mesh,colors_to_label,fdi_to_label
from lightning.pytorch import seed_everything
import copy
from scipy import spatial

# same function in mesh_dataset
def process_mesh(mesh: trimesh, labels: torch.tensor = None):
    mesh_faces = torch.from_numpy(mesh.faces.copy()).float()
    mesh_triangles = torch.from_numpy(mesh.vertices[mesh.faces]).float()
    mesh_face_normals = torch.from_numpy(mesh.face_normals.copy()).float()
    mesh_vertices_normals = torch.from_numpy(mesh.vertex_normals[mesh.faces]).float()
    if labels is None:
        labels = torch.from_numpy(colors_to_label(mesh.visual.face_colors.copy())).long()
    return mesh_faces, mesh_triangles, mesh_vertices_normals, mesh_face_normals, labels

# similar function as PreTransform in preprocessing.py
def preporces(data):
    mesh_faces, mesh_triangles, mesh_vertices_normals, mesh_face_normals, labels = data
    mesh = trimesh.Trimesh(**trimesh.triangles.to_kwargs(mesh_triangles.cpu().detach().numpy()))

    points = torch.from_numpy(mesh.vertices)
    v_normals = torch.from_numpy(mesh.vertex_normals)

    s, _ = mesh_faces.size()
    x = torch.zeros(s, 24).float()
    x[:, :3] = mesh_triangles[:, 0]
    x[:, 3:6] = mesh_triangles[:, 1]
    x[:, 6:9] = mesh_triangles[:, 2]
    x[:, 9:12] = mesh_triangles.mean(dim=1)
    x[:, 12:15] = mesh_vertices_normals[:, 0]
    x[:, 15:18] = mesh_vertices_normals[:, 1]
    x[:, 18:21] = mesh_vertices_normals[:, 2]
    x[:, 21:] = mesh_face_normals

    maxs = points.max(dim=0)[0]
    mins = points.min(dim=0)[0]
    means = points.mean(axis=0)
    stds = points.std(axis=0)
    nmeans = v_normals.mean(axis=0)
    nstds = v_normals.std(axis=0)
    nmeans_f = mesh_face_normals.mean(axis=0)
    nstds_f = mesh_face_normals.std(axis=0)
    for i in range(3):
        # normalize coordinate
        x[:, i] = (x[:, i] - means[i]) / stds[i]  # point 1
        x[:, i + 3] = (x[:, i + 3] - means[i]) / stds[i]  # point 2
        x[:, i + 6] = (x[:, i + 6] - means[i]) / stds[i]  # point 3
        x[:, i + 9] = (x[:, i + 9] - mins[i]) / (maxs[i] - mins[i])  # centre
        # normalize normal vector
        x[:, i + 12] = (x[:, i + 12] - nmeans[i]) / nstds[i]  # normal1
        x[:, i + 15] = (x[:, i + 15] - nmeans[i]) / nstds[i]  # normal2
        x[:, i + 18] = (x[:, i + 18] - nmeans[i]) / nstds[i]  # normal3
        x[:, i + 21] = (x[:, i + 21] - nmeans_f[i]) / nstds_f[i]  # face normal

    pos = x[:, 9:12]

    return pos, x, labels

# same function(method) in mesh_dataset.Teeth3DSDataset
def Downsample(mesh,labels):
    mesh_simplifier = pyfqmr.Simplify()
    mesh_simplifier.setMesh(mesh.vertices, mesh.faces)
    mesh_simplifier.simplify_mesh(target_count=16000, aggressiveness=3, preserve_border=True, verbose=0,
                                  max_iterations=2000)
    new_positions, new_face, _ = mesh_simplifier.getMesh()
    mesh_simple = trimesh.Trimesh(vertices=new_positions, faces=new_face)
    vertices = mesh_simple.vertices
    faces = mesh_simple.faces
    if faces.shape[0] < 16000:
        fs_diff = 16000 - faces.shape[0]
        faces = np.append(faces, np.zeros((fs_diff, 3), dtype="int"), 0)
    elif faces.shape[0] > 16000:
        mesh_simple = trimesh.Trimesh(vertices=vertices, faces=faces)
        samples, face_index = trimesh.sample.sample_surface_even(mesh_simple, 16000)
        mesh_simple = trimesh.Trimesh(vertices=mesh_simple.vertices, faces=mesh_simple.faces[face_index])
        faces = mesh_simple.faces
        vertices = mesh_simple.vertices
    mesh_simple = trimesh.Trimesh(vertices=vertices, faces=faces)

    mesh_v_mean = mesh.vertices[mesh.faces].mean(axis=1)
    mesh_simple_v = mesh_simple.vertices
    tree = spatial.KDTree(mesh_v_mean)
    query = mesh_simple_v[faces].mean(axis=1)
    distance, index = tree.query(query)
    labels = labels[index].flatten()
    return mesh_simple,labels

# reverse normalization
def PostProces(data_OG_def,x_def):
    _, mesh_triangles, _, mesh_face_normals, _ = data_OG_def
    mesh = trimesh.Trimesh(**trimesh.triangles.to_kwargs(mesh_triangles.cpu().detach().numpy()))
       
    maxs = mesh.vertices.max(axis=0)
    mins =  mesh.vertices.min(axis=0)
    means =  mesh.vertices.mean(axis=0)
    stds =  mesh.vertices.std(axis=0)
    nmeans = mesh.vertex_normals.mean(axis=0)
    nstds = mesh.vertex_normals.std(axis=0)
    nmeans_f = mesh_face_normals.mean(axis=0)
    nstds_f = mesh_face_normals.std(axis=0)
    for i in range(3):
        #  coordinate
        x_def[:, i] = (x_def[:, i] + means[i]) * stds[i]  # point 1
        x_def[:, i + 3] = (x_def[:, i + 3] + means[i]) * stds[i]  # point 2
        x_def[:, i + 6] = (x_def[:, i + 6] + means[i]) * stds[i]  # point 3
        x_def[:, i + 9] = (x_def[:, i + 9] + mins[i]) * (maxs[i] - mins[i])  # centre
        #  normal vector
        x_def[:, i + 12] = (x_def[:, i + 12] + nmeans[i]) * nstds[i]  # normal1
        x_def[:, i + 15] = (x_def[:, i + 15] + nmeans[i]) * nstds[i]  # normal2
        x_def[:, i + 18] = (x_def[:, i + 18] + nmeans[i]) * nstds[i]  # normal3
        x_def[:, i + 21] = (x_def[:, i + 21] + nmeans_f[i]) * nstds_f[i]  # face normal
    return x_def

SEED = 42
use_gpu=True
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
torch.set_float32_matmul_precision('medium')
random.seed(SEED)
seed_everything(SEED, workers=True)

#-----Define values
Model_Teeth=r'\\.obj' # .obj file path in Teeth3DS dataset example: Teeth3DS\Upper\\0JN50XQR\\0JN50XQR_upper.obj
ML_parameters='\\.ckpt' # model parameter file

#----------Model----------
model = LitDilatedToothSegmentationNetwork.load_from_checkpoint(ML_parameters)
if use_gpu==True:
   model = model.cuda()
   
#----Import model
mesh=trimesh.load(Path(Model_Teeth))
with open(Model_Teeth.replace('.obj', '.json')) as f:
     data = json.load(f)
labels = np.array(data["labels"])
labels = labels[mesh.faces]
labels = labels[:, 0]
labels = fdi_to_label(labels)

#----Downsample
mesh_simple,labels=Downsample(mesh,labels)

#----Preporcess
data = process_mesh(mesh_simple, torch.from_numpy(labels).long())
data_OG=copy.copy(data)
data =preporces(data)

#----Ground truth model labels
ground_truth = data[2]
mesh_gt = color_mesh(mesh_simple, ground_truth.numpy())
# mesh_gt.export('gt.ply') # export ground truth 3D model

#----Use model
pre_labels = model.predict_labels(data).cpu().numpy()
x=PostProces(data_OG,data[1]) # Postprocess

triangles = x[:, :9].reshape(-1, 3, 3)
mesh = trimesh.Trimesh(**trimesh.triangles.to_kwargs(triangles.cpu().detach().numpy()))
mesh_pred = color_mesh(mesh, pre_labels)
# mesh_pred.export('pred.ply') # export predicted 3D model

#----Show models with highlighted teeths. Requare polyscope (https://github.com/nmwsharp/polyscope) to be installed
ps.init()
color_groud=ps.register_surface_mesh('Original', mesh_simple.vertices-mesh_simple.centroid, mesh_simple.faces)
color_groud.add_color_quantity("groud labels", mesh_gt.visual.face_colors[:,:3]/255, defined_on='faces')
color_pred=ps.register_surface_mesh('Final model', mesh_pred.vertices-mesh_pred.centroid, mesh_pred.faces)
color_pred.add_color_quantity("predicted labels", mesh_pred.visual.face_colors[:,:3]/255, defined_on='faces')
ps.show()
ps.remove_all_structures()

