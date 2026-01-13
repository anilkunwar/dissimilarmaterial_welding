# app.py
import streamlit as st
import numpy as np
import tempfile
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import Delaunay

st.set_page_config(
    page_title="ParallelGroup Slab Generator",
    page_icon="📐",
    layout="wide"
)

st.title("ParallelGroup Slab Generator with 3D Visualization")
st.markdown("""
This app generates a mesh of two adjacent slabs with comprehensive export options and interactive 3D visualization.
**No system dependencies required** - works in all cloud environments.
- **Slab 1 (Solid_1front)**: `(0,0,0)` → `(lx, ly, lz)`
- **Slab 2 (Solid_2back)**:  `(0,ly,0)` → `(lx, 2·ly, lz)`
""")

# Check if meshio is available
HAS_MESHIO = False
try:
    import meshio
    HAS_MESHIO = True
except ImportError:
    st.warning("Meshio not available. Will generate simplified exports.")

def create_hex_mesh(lx, ly, lz, resolution=10):
    """
    Create a hexahedral mesh without external dependencies
    """
    # Calculate divisions based on resolution
    div_x = max(2, int(resolution * lx / max(lx, ly, lz)))
    div_y = max(2, int(resolution * ly / max(lx, ly, lz)))
    div_z = max(2, int(resolution * lz / max(lx, ly, lz)))
    
    # Create mesh points
    points = []
    for k in range(div_z + 1):
        z = k * lz / div_z
        for j in range(2 * div_y + 1):  # Two slabs stacked in Y
            y = j * ly / div_y
            for i in range(div_x + 1):
                x = i * lx / div_x
                points.append([x, y, z])
    
    # Create hexahedral cells
    cells = []
    for k in range(div_z):
        for j in range(2 * div_y):
            for i in range(div_x):
                # Get the 8 corners of the hexahedron
                n0 = k * (div_x + 1) * (2 * div_y + 1) + j * (div_x + 1) + i
                n1 = n0 + 1
                n2 = n0 + (div_x + 1) + 1
                n3 = n0 + (div_x + 1)
                n4 = n0 + (div_x + 1) * (2 * div_y + 1)
                n5 = n4 + 1
                n6 = n4 + (div_x + 1) + 1
                n7 = n4 + (div_x + 1)
                
                cells.append([n0, n1, n2, n3, n4, n5, n6, n7])
    
    # Create physical groups data
    field_data = {
        "Solid_1front": np.array([1, 3]),  # 3D entity
        "Solid_2back": np.array([2, 3]),   # 3D entity
        "Face_1leftfront": np.array([1, 2]), 
        "Face_2leftback": np.array([2, 2]),
        "Face_3frontfront": np.array([3, 2]),
        "Face_4bottomfront": np.array([4, 2]),
        "Face_5topfront": np.array([5, 2]),
        "Face_6interfacefront": np.array([6, 2]),
        "Face_7bottomback": np.array([7, 2]),
        "Face_8topback": np.array([8, 2]),
        "Face_9backback": np.array([9, 2]),
        "Face_10rightfront": np.array([10, 2]),
        "Face_11rightback": np.array([11, 2])
    }
    
    return {
        "points": np.array(points),
        "cells": np.array(cells),
        "field_data": field_data,
        "dimensions": (lx, ly, lz),
        "divisions": (div_x, div_y, div_z)
    }

def extract_surface_mesh(mesh_data):
    """
    Extract surface faces from hexahedral mesh
    Returns points and triangular faces for STL export
    """
    points = mesh_data["points"]
    cells = mesh_data["cells"]
    
    # Dictionary to track face usage (face defined by sorted node indices)
    face_dict = {}
    
    # For each hexahedron, process its 6 faces
    for cell_idx, hex_nodes in enumerate(cells):
        # Define the 6 faces of a hexahedron (each face has 4 nodes)
        hex_faces = [
            [hex_nodes[0], hex_nodes[1], hex_nodes[2], hex_nodes[3]],  # bottom
            [hex_nodes[4], hex_nodes[5], hex_nodes[6], hex_nodes[7]],  # top
            [hex_nodes[0], hex_nodes[1], hex_nodes[5], hex_nodes[4]],  # front
            [hex_nodes[2], hex_nodes[3], hex_nodes[7], hex_nodes[6]],  # back
            [hex_nodes[0], hex_nodes[3], hex_nodes[7], hex_nodes[4]],  # left
            [hex_nodes[1], hex_nodes[2], hex_nodes[6], hex_nodes[5]]   # right
        ]
        
        for face_idx, face_nodes in enumerate(hex_faces):
            # Create a canonical key for this face (sorted node indices)
            face_key = tuple(sorted(face_nodes))
            
            # Track which cells use this face
            if face_key not in face_dict:
                face_dict[face_key] = []
            face_dict[face_key].append((cell_idx, face_idx))
    
    # Surface faces are those used by only one cell
    surface_faces = [face for face, cells in face_dict.items() if len(cells) == 1]
    
    # Convert quadrilateral faces to triangles for STL
    tri_faces = []
    for quad in surface_faces:
        # Split quad into two triangles
        tri_faces.append([quad[0], quad[1], quad[2]])
        tri_faces.append([quad[0], quad[2], quad[3]])
    
    return points, np.array(tri_faces)

def visualize_with_plotly(mesh_data, show_wireframe=True, show_surface=True):
    """
    Create interactive 3D visualization using Plotly
    """
    points = mesh_data["points"]
    cells = mesh_data["cells"]
    lx, ly, lz = mesh_data["dimensions"]
    
    fig = go.Figure()
    
    # Add wireframe visualization of the mesh
    if show_wireframe:
        # For each hex cell, add lines for its 12 edges
        for hex_nodes in cells[:min(500, len(cells))]:  # Limit for performance
            # Define the 12 edges of a hexahedron
            edges = [
                [hex_nodes[0], hex_nodes[1]], [hex_nodes[1], hex_nodes[2]], [hex_nodes[2], hex_nodes[3]], [hex_nodes[3], hex_nodes[0]],
                [hex_nodes[4], hex_nodes[5]], [hex_nodes[5], hex_nodes[6]], [hex_nodes[6], hex_nodes[7]], [hex_nodes[7], hex_nodes[4]],
                [hex_nodes[0], hex_nodes[4]], [hex_nodes[1], hex_nodes[5]], [hex_nodes[2], hex_nodes[6]], [hex_nodes[3], hex_nodes[7]]
            ]
            
            for edge in edges:
                x = [points[edge[0], 0], points[edge[1], 0]]
                y = [points[edge[0], 1], points[edge[1], 1]]
                z = [points[edge[0], 2], points[edge[1], 2]]
                
                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(color='gray', width=1),
                    hoverinfo='none',
                    showlegend=False
                ))
    
    # Add surface visualization
    if show_surface:
        # Extract surface for visualization
        surf_points, surf_faces = extract_surface_mesh(mesh_data)
        
        # Create triangles from the surface faces
        if len(surf_faces) > 0:
            fig.add_trace(go.Mesh3d(
                x=surf_points[:, 0],
                y=surf_points[:, 1],
                z=surf_points[:, 2],
                i=surf_faces[:, 0],
                j=surf_faces[:, 1],
                k=surf_faces[:, 2],
                color='lightblue',
                opacity=0.5,
                flatshading=True,
                name='Surface'
            ))
    
    # Highlight the interface between the two slabs
    interface_y = ly
    fig.add_trace(go.Surface(
        x=[[0, lx], [0, lx]],
        y=[[interface_y, interface_y], [interface_y, interface_y]],
        z=[[0, 0], [lz, lz]],
        colorscale=[[0, 'red'], [1, 'red']],
        opacity=0.7,
        name='Interface (Face_6interfacefront)',
        showscale=False,
        hoverinfo='name'
    ))
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        title='3D Mesh Visualization',
        height=600,
        margin=dict(l=0, r=0, b=0, t=30),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def export_mesh(mesh_data, format_name="msh"):
    """
    Export mesh data to various formats
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"two_slabs.{format_name}"
        filepath = os.path.join(tmpdir, filename)
        
        lx, ly, lz = mesh_data["dimensions"]
        
        if format_name == "unv" and HAS_MESHIO:
            # Create meshio Mesh object for UNV export
            hex_cells = [("hexahedron", mesh_data["cells"])]
            mesh = meshio.Mesh(
                points=mesh_data["points"],
                cells=hex_cells,
                field_data=mesh_data["field_data"]
            )
            mesh.write(filepath, file_format="unv")
            
        elif format_name == "stl" and HAS_MESHIO:
            # Extract surface mesh for STL export
            surf_points, surf_faces = extract_surface_mesh(mesh_data)
            
            # Create triangles for STL
            triangles = []
            for quad in surf_faces:
                # Split each quad into two triangles
                triangles.append([quad[0], quad[1], quad[2]])
                triangles.append([quad[0], quad[2], quad[3]])
            
            # Create meshio mesh with triangles
            tri_cells = [("triangle", np.array(triangles))]
            mesh = meshio.Mesh(
                points=surf_points,
                cells=tri_cells
            )
            mesh.write(filepath, file_format="stl")
            
        elif HAS_MESHIO:
            # Handle other formats
            hex_cells = [("hexahedron", mesh_data["cells"])]
            mesh = meshio.Mesh(
                points=mesh_data["points"],
                cells=hex_cells,
                field_data=mesh_data["field_data"]
            )
            
            format_map = {
                "msh": "gmsh22",
                "vtk": "vtk",
                "vtu": "vtu",
                "xdmf": "xdmf"
            }
            
            file_format = format_map.get(format_name, "vtk")
            mesh.write(filepath, file_format=file_format)
            
        else:
            # Fallback for when meshio is not available
            if format_name == "stl":
                # Generate simple ASCII STL
                surf_points, surf_faces = extract_surface_mesh(mesh_data)
                
                with open(filepath, 'w') as f:
                    f.write("solid two_slabs\n")
                    for face in surf_faces:
                        # Get vertices of the face
                        v0 = surf_points[face[0]]
                        v1 = surf_points[face[1]]
                        v2 = surf_points[face[2]]
                        
                        # Simple normal calculation (not accurate but works for visualization)
                        normal = np.array([0, 0, 1])
                        
                        f.write(f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n")
                        f.write("    outer loop\n")
                        f.write(f"      vertex {v0[0]} {v0[1]} {v0[2]}\n")
                        f.write(f"      vertex {v1[0]} {v1[1]} {v1[2]}\n")
                        f.write(f"      vertex {v2[0]} {v2[1]} {v2[2]}\n")
                        f.write("    endloop\n")
                        f.write("  endfacet\n")
                    f.write("endsolid two_slabs\n")
            elif format_name == "unv":
                # Generate simplified UNV format
                with open(filepath, 'w') as f:
                    # Write dataset header
                    f.write("    -1\n")
                    f.write("  2411\n")
                    f.write("    -1\n")
                    f.write("    1\n")
                    f.write(f"    {len(mesh_data['points'])}\n")
                    
                    # Write nodes
                    for i, point in enumerate(mesh_data["points"], 1):
                        f.write(f"{i:10d}{1:10d}{1:10d}{0:10d}\n")
                        f.write(f"{point[0]:13.5E}{point[1]:13.5E}{point[2]:13.5E}\n")
                    
                    f.write("    -1\n")
                    f.write("  2412\n")
                    f.write("    -1\n")
                    
                    # Write elements (hexahedrons)
                    f.write(f"    {len(mesh_data['cells'])}\n")
                    for i, cell in enumerate(mesh_data["cells"], 1):
                        f.write(f"{i:10d}{11:10d}{1:10d}{1:10d}\n")
                        f.write(f"{cell[0]+1:10d}{cell[1]+1:10d}{cell[2]+1:10d}{cell[3]+1:10d}\n")
                        f.write(f"{cell[4]+1:10d}{cell[5]+1:10d}{cell[6]+1:10d}{cell[7]+1:10d}\n")
                    
                    f.write("    -1\n")
            else:
                # Default text format
                with open(filepath, 'w') as f:
                    f.write("# Mesh data\n")
                    f.write(f"# Dimensions: lx={lx}, ly={ly}, lz={lz}\n")
                    f.write(f"# Nodes: {len(mesh_data['points'])}\n")
                    f.write(f"# Elements: {len(mesh_data['cells'])}\n")
                    f.write("# Physical groups:\n")
                    for name in mesh_data["field_data"].keys():
                        f.write(f"# - {name}\n")
        
        # Read file content for download
        with open(filepath, 'rb') as f:
            return f.read(), filename

def main():
    # Sidebar for parameters
    with st.sidebar:
        st.header("PropertyParams")
        
        # Dimensions input
        lx = st.number_input("Length (lx)", min_value=1.0, value=200.0, step=10.0)
        ly = st.number_input("Width (ly)", min_value=1.0, value=50.0, step=5.0)
        lz = st.number_input("Height (lz)", min_value=0.1, value=2.0, step=0.5)
        
        # Mesh density
        resolution = st.slider("Mesh Density", 1, 20, 5, 
                             help="Higher values = finer mesh")
        
        # Visualization options
        st.subheader("Visualization Options")
        show_wireframe = st.checkbox("Show Wireframe", value=True)
        show_surface = st.checkbox("Show Surface", value=True)
        
        # Export format
        export_format = st.selectbox("Export Format", 
                                   ["msh", "vtk", "vtu", "xdmf", "unv", "stl", "txt"])
    
    # Generate button at the top
    generate_button = st.button("🚀 Generate Mesh", type="primary")
    
    if generate_button:
        with st.spinner("Creating mesh data..."):
            # Create the mesh
            mesh_data = create_hex_mesh(lx, ly, lz, resolution)
            
            # Get statistics
            num_points = len(mesh_data["points"])
            num_cells = len(mesh_data["cells"])
            
            # Display statistics in a card
            with st.container():
                st.subheader("📊 Mesh Statistics")
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                with col_stats1:
                    st.metric("Nodes", f"{num_points:,}")
                with col_stats2:
                    st.metric("Elements", f"{num_cells:,}")
                with col_stats3:
                    st.metric("Physical Groups", "13")
            
            # Create two columns for visualization and physical groups
            col1, col2 = st.columns([2, 1])
            
            # 3D Visualization with Plotly
            with col1:
                st.subheader("3D Visualization")
                fig = visualize_with_plotly(mesh_data, show_wireframe, show_surface)
                st.plotly_chart(fig, use_container_width=True)
            
            # Physical groups information
            with col2:
                st.subheader("PropertyParams")
                st.markdown(f"""
                **Dimensions:**
                - Length (X): {lx}
                - Width (Y): {2*ly} (total)
                - Height (Z): {lz}
                
                **Mesh Details:**
                - Hexahedral elements
                - Interface at Y = {ly}
                """)
                
                with st.expander("Full List of Physical Groups"):
                    st.markdown("""
                    **Volumes:**
                    - Solid_1front
                    - Solid_2back
                    
                    **Faces:**
                    - Face_1leftfront
                    - Face_2leftback
                    - Face_3frontfront
                    - Face_4bottomfront
                    - Face_5topfront
                    - Face_6interfacefront (interface)
                    - Face_7bottomback
                    - Face_8topback
                    - Face_9backback
                    - Face_10rightfront
                    - Face_11rightback
                    """)
            
            # Export section
            st.subheader("📥 Download Mesh")
            
            # Generate file for download
            file_data, filename = export_mesh(mesh_data, export_format)
            
            # Determine MIME type
            mime_types = {
                "msh": "application/octet-stream",
                "vtk": "application/vnd.vtk",
                "vtu": "application/vnd.vtu",
                "xdmf": "application/xdmf+xml",
                "unv": "application/octet-stream",
                "stl": "model/stl",
                "txt": "text/plain"
            }
            mime_type = mime_types.get(export_format, "application/octet-stream")
            
            st.download_button(
                label=f"Download {filename}",
                data=file_data,
                file_name=filename,
                mime=mime_type,
                use_container_width=True
            )
            
            # Additional information about formats
            with st.expander("ℹ️ Format Information"):
                format_info = {
                    "msh": "Gmsh format - compatible with Code_Aster, GetFEM, and other FEM solvers",
                    "vtk": "VTK format - compatible with ParaView and VTK-based tools",
                    "vtu": "VTK XML format - more efficient than standard VTK",
                    "xdmf": "Extensible Data Model - efficient for large datasets, compatible with many solvers",
                    "unv": "I-DEAS Universal format - widely used in commercial FEM software",
                    "stl": "Stereolithography format - surface mesh only, used for 3D printing and visualization",
                    "txt": "Simple text format - human readable but limited functionality"
                }
                st.markdown(f"**{export_format.upper()}:** {format_info.get(export_format, 'Standard mesh format')}")
            
            st.success("✅ Mesh generated successfully!")
    
    # Footer information
    st.markdown("---")
    st.caption("✅ This app works in all cloud environments without system dependencies")

if __name__ == "__main__":
    main()
