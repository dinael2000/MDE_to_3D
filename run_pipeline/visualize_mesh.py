import os

import plotly.graph_objects as go
import numpy as np
import open3d as o3d

def visualize_meshes(vertices,triangles,vertex_colors):
    """
    A function that visualizes a given mesh
    using the plotly library
    """
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                vertexcolor=vertex_colors,
                opacity=1.0
            )
        ],
        layout=dict(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data"
            )
        )
    )
    fig.show()


def batch_visualize_meshes(mesh_folder):
    """
    A function that batch processes
    meshes to be visualized using 
    the visualize_meshes function
    """
    # Reads mesh folder
    mesh_folder_list = os.listdir(mesh_folder)

    if len(mesh_folder_list) == 0: 
        print("Directory does not contain files")
        return

    # Reads mesh files
    for mesh_file in mesh_folder_list:
        mesh_path = os.path.join(mesh_folder, mesh_file)

        mesh = o3d.io.read_triangle_mesh(mesh_path)

        if mesh.is_empty():
            print(f"Empty mesh: {mesh_file}")
            continue
        # Reads mesh vertices, triangles, colors
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
            
        if not mesh.has_triangle_normals():
            mesh.compute_triangle_normals()

        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)

        if not mesh.has_vertex_colors():
            print("Mesh has no vertex colors")
            continue
        vertex_colors = np.asarray(mesh.vertex_colors)
        vertex_colors = (vertex_colors * 255).astype(np.uint8)

        # Runs visualize_meshes function
        visualize_meshes(vertices, triangles, vertex_colors)
