
import trimesh
import torch
import numpy as np
import polyscope as ps
from models.dilated_tooth_seg_network import LitDilatedToothSegmentationNetwork
from sklearn.decomposition import PCA
import random
from teeth_numbering import color_mesh,_teeth_labels,_teeth_color

import fast_simplification #!!!!
from lightning.pytorch import seed_everything
import copy
from scipy.spatial import KDTree

def process_mesh(mesh: trimesh, labels: torch.tensor = None):
    mesh_faces = torch.from_numpy(mesh.faces.copy()).float()
    mesh_triangles = torch.from_numpy(mesh.vertices[mesh.faces]).float()
    mesh_face_normals = torch.from_numpy(mesh.face_normals.copy()).float()
    mesh_vertices_normals = torch.from_numpy(mesh.vertex_normals[mesh.faces]).float()
    return mesh_faces, mesh_triangles, mesh_vertices_normals, mesh_face_normals, labels

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

def get_pca_rotation(mesh):
    vertices = mesh.vertices
    pca = PCA(n_components=3)
    pca.fit(vertices)
    rotation_matrix = pca.components_
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:, 2] = -rotation_matrix[:, 2]  # Adjusting the z-axis
    centroid = np.mean(vertices, axis=0)
    return rotation_matrix, centroid

def align_meshes(mesh,src_rot_matrix,src_centroid,tgt_rot_matrix,tgt_centroid):
    # Compute the transformation matrix to align source to target
    transformation_matrix = src_rot_matrix.T @ tgt_rot_matrix

    # Apply  rotation
    rotated_vertices = mesh.vertices.dot(transformation_matrix)
    rotated_mesh = trimesh.Trimesh(vertices=rotated_vertices, faces=mesh.faces)

    # Translate source centroid to target centroid
    translation_vector = tgt_centroid - np.mean(rotated_mesh.vertices, axis=0)
    rotated_mesh.vertices += translation_vector
    return rotated_mesh

def Downsample(mesh): 
    points_out, faces_out, collapses = fast_simplification.simplify(mesh.vertices, mesh.faces,(1-16000/mesh.faces.shape[0]) , return_collapses=True)
    points_out, faces_out, indice_mapping = fast_simplification.replay_simplification(mesh.vertices.astype('float32') , mesh.faces.astype('float32') , collapses.astype('int32'))
    mesh_simple = trimesh.Trimesh(vertices=points_out, faces=faces_out)
    
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
    labels=np.zeros(faces.shape[0])
    return mesh_simple,labels,indice_mapping

def Erosion(Extr_part_def,how_much_to_reduce=2):
     for lo_none in range(how_much_to_reduce): # reguliuoti kiek pasalinama
         Face_indices=np.arange(Extr_part_def.faces.shape[0])
         Boundary_edges_segment=Extr_part_def.edges[trimesh.grouping.group_rows(Extr_part_def.edges_sorted, require_count=1)]
         Boundary_edges_segment=np.unique(Boundary_edges_segment.flatten())
         mask=np.unique(Extr_part_def.vertex_faces[Boundary_edges_segment,:])[1:]     
         Extr_part_def.update_faces(Face_indices[~np.isin(Face_indices,mask)])
         Extr_part_def.remove_unreferenced_vertices()
       
         Extr_part_splited=Extr_part_def.split(only_watertight=False)
       
         Extr_part_def_number=np.argmax([i.area for i in Extr_part_splited])
         Extr_part_splited=Extr_part_splited[Extr_part_def_number]
        
     return Extr_part_splited   
  
def Corect_plane(mesh_simple,src_rot_matrix, src_centroid): 
# Ideja: rasti okliucija apibreziancias dantu virsuniu noramaliu vidurkiu.
# Jei normales elemetai visi taigiami-kaip ir didzioji dalis dantu lanku duonbazeje tada nenaudojam flipinimo. 
     mesh_simple_first= align_meshes(mesh_simple,src_rot_matrix, src_centroid,tgt_rot_matrix,tgt_centroid)
     
     #_, _, max_pv_15, min_pv_15 = igl.principal_curvature(mesh_simple_first.vertices, mesh_simple_first.faces, radius=5)
     mean_curv=trimesh.curvature.discrete_mean_curvature_measure(mesh_simple, mesh_simple.vertices, 0.01)
     disc_means=np.tanh(mean_curv*np.arctanh((2**8 - 2) / (2**8 - 1)))
     return np.all(mesh_simple_first.vertex_normals[np.where(disc_means>0.7)].mean(axis=0)>0)

#-----------Stable rotaion values for aligning to dataset -------------------
tgt_rot_matrix =np.array([[ 0.99481732,  0.08303923, -0.05867689],
                          [-0.0920596 ,  0.98060088, -0.17305184],
                          [ 0.04316852,  0.17755674,  0.9831633 ]])

tgt_centroid=np.array([[2.03561511,  -0.65064242, -90.05015842]])
#-------------------------------------------

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
#----------Model----------
model = LitDilatedToothSegmentationNetwork.load_from_checkpoint('ML_param.ckpt')

if use_gpu==True:
   model = model.cuda()
   
#----Import model
mesh=trimesh.load('D:/MANO/Prodentum_darbai/Wear_compare/Fixed/Medit_upper.stl')
mesh_OG=copy.copy(mesh)
#----Downsample
mesh_simple,labels,corespondance=Downsample(mesh)

#----Correct aligment
src_rot_matrix, src_centroid = get_pca_rotation(mesh_simple)
Is_not_fliped=Corect_plane(mesh_simple,src_rot_matrix, src_centroid)

if Is_not_fliped:
    mesh_simple= align_meshes(mesh_simple,src_rot_matrix, src_centroid,tgt_rot_matrix,tgt_centroid)
else:
    flip_y_matrix = np.array([[-1, 0, 0],[0, 1, 0],[0, 0, -1]])
    tgt_rot_matrix = np.dot(tgt_rot_matrix, flip_y_matrix)
    mesh_simple= align_meshes(mesh_simple,src_rot_matrix, src_centroid,tgt_rot_matrix,tgt_centroid)

#----Preporcess
data = process_mesh(mesh_simple, torch.from_numpy(labels).long())
data_OG=copy.copy(data)

data =preporces(data)

#----Use model
ground_truth = data[2]
pre_labels = model.predict_labels(data).cpu().numpy()

x=PostProces(data_OG,data[1]) # Postprocess

triangles = x[:, :9].reshape(-1, 3, 3)
mesh = trimesh.Trimesh(**trimesh.triangles.to_kwargs(triangles.cpu().detach().numpy()))
mesh= align_meshes(mesh,tgt_rot_matrix,tgt_centroid,src_rot_matrix, src_centroid) # Back to original
mesh_simple=align_meshes(mesh_simple,tgt_rot_matrix,tgt_centroid,src_rot_matrix, src_centroid) 

mesh_pred = color_mesh(mesh, pre_labels)

# mesh_pred.export(f'pred.ply')
# mesh_gt.export(f'gt.ply')

#----Show models with highlighted teeths
# ps.init()
# ps.register_surface_mesh('mesh_simple', mesh_simple.vertices, mesh_simple.faces)
# ps.register_surface_mesh('Original', mesh_OG.vertices, mesh_OG.faces)
# color=ps.register_surface_mesh('Predicting', mesh_pred.vertices, mesh_pred.faces)
# color.add_color_quantity("rand colors2", mesh_pred.visual.face_colors[:,:3]/255, defined_on='faces')
# ps.show()
# ps.remove_all_structures()

#----Extract teeth from original mesh
# """
test_mesh_predicted=copy.copy(mesh_pred)
Store_segments=[]
Which_teeth_store=[]

for i in range(len(np.unique(pre_labels))):
    first_unique_color = np.unique(test_mesh_predicted.visual.face_colors, axis=0)[i]
    matches_first_color = np.where(np.all(test_mesh_predicted.visual.face_colors == first_unique_color, axis=1))[0]

    Vert_decimated=np.unique(test_mesh_predicted.faces[matches_first_color])
    
    tree = KDTree(mesh_simple.vertices)
    Vert_decimated_true=[]
    for ids in Vert_decimated:
        _, indices = tree.query(test_mesh_predicted.vertices[ids], k=1)
        Vert_decimated_true.append(indices)
 
    Where_are_they=np.where(np.isin(corespondance,Vert_decimated_true))[0]
    
    mask = np.isin(mesh_OG.faces, Where_are_they)
    faces_with_vertices_mask = mask.any(axis=1)
    unique_faces = np.where(faces_with_vertices_mask)[0]
    unique_faces=np.unique(np.hstack(unique_faces))
     
    Extr_part=mesh_OG.submesh([unique_faces])[0] # create submesh
    
    #-------------REMOVE some triangles (erosion)
    Which_teeth=[i for i in np.unique(pre_labels) if np.all(first_unique_color[:3]==_teeth_color[i])][0]
        
    if Which_teeth!=0:
        Extr_part=Erosion(Extr_part)
    
    Extr_part.metadata['file_name']=_teeth_labels[Which_teeth] # changing names
    Extr_part.metadata['color']=np.array(_teeth_color[Which_teeth])/255
    
    Store_segments.append(Extr_part)
    
    # ps.init()
    # ps.register_surface_mesh(_teeth_labels[Which_teeth], Extr_part.vertices, Extr_part.faces,color=[1,0,0])
    # Art=ps.register_surface_mesh('Original_arch',mesh_OG.vertices,mesh_OG.faces,color=[1,1,1])
    # Art.add_scalar_quantity("SDF", Ext_scal,vminmax=(-0.3, 0.3))
    # ps.register_surface_mesh(_teeth_labels[Which_teeth]+'_original_donwsampled', mesh_sub_par[0].vertices, mesh_sub_par[0].faces,enabled=False)
    # ps.register_point_cloud('Choosen_vertices',Vertices_trim)
    
    # ps.show()
    # ps.remove_all_structures()

    
ps.init()
ps.register_surface_mesh('z_Origina;', mesh_OG.vertices, mesh_OG.faces)
ps.register_surface_mesh('z_Origina_reduced;', mesh_simple.vertices, mesh_simple.faces)
for jj in Store_segments:
    ps.register_surface_mesh(jj.metadata['file_name'], jj.vertices, jj.faces,color=jj.metadata['color'],enabled =True)
ps.show()
ps.remove_all_structures()

