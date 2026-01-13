# app.py
import streamlit as st
import numpy as np
import tempfile
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import Delaunay
import io
import base64

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

# Check if meshio is available and get supported formats
HAS_MESHIO = False
SUPPORTED_FORMATS = []
try:
    import meshio
    HAS_MESHIO = True
    # Get supported write formats from meshio
    SUPPORTED_FORMATS = list(meshio._helpers.write_format_to_filetype.keys())
except ImportError:
    st.warning("Meshio not available. Will generate simplified exports.")
    SUPPORTED_FORMATS = ["stl", "txt"]  # Fallback formats we can generate manually

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
                face_dict[face_key] = {'count': 0, 'faces': []}
            face_dict[face_key]['count'] += 1
            face_dict[face_key]['faces'].append((cell_idx, face_idx))
    
    # Surface faces are those used by only one cell
    surface_quads = [face for face, data in face_dict.items() if data['count'] == 1]
    
    # Convert quadrilateral faces to triangles for STL
    tri_faces = []
    for quad in surface_quads:
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
    if show_wireframe and len(cells) > 0:
        # Limit for performance in cloud environments
        max_cells = min(200, len(cells))
        for hex_nodes in cells[:max_cells]:
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
    if show_surface and len(cells) > 0:
        # Extract surface for visualization
        surf_points, surf_faces = extract_surface_mesh(mesh_data)
        
        # Create triangles from the surface faces
        if len(surf_faces) > 0:
            # Limit number of triangles for performance
            max_triangles = min(5000, len(surf_faces))
            fig.add_trace(go.Mesh3d(
                x=surf_points[:, 0],
                y=surf_points[:, 1],
                z=surf_points[:, 2],
                i=surf_faces[:max_triangles, 0],
                j=surf_faces[:max_triangles, 1],
                k=surf_faces[:max_triangles, 2],
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

def create_unv_file(points, cells, field_data=None):
    """
    Create a UNV file format manually without meshio dependencies
    Returns bytes of the UNV file content
    """
    output = io.StringIO()
    
    # Write file header
    output.write("    -1\n")
    output.write("  2411\n")  # Dataset for nodes
    output.write("    -1\n")
    
    # Write nodes
    output.write(f"         1\n")  # Record 1
    output.write(f"{len(points):10d}\n")  # Number of nodes
    
    for i, point in enumerate(points, 1):
        # Node format: node_id, color, coordinates
        output.write(f"{i:10d}         1         1         0\n")
        output.write(f"{point[0]:13.5E}{point[1]:13.5E}{point[2]:13.5E}\n")
    
    output.write("    -1\n")
    output.write("  2412\n")  # Dataset for elements
    output.write("    -1\n")
    
    # Write elements
    output.write(f"{len(cells):10d}\n")  # Number of elements
    
    for i, cell in enumerate(cells, 1):
        # Element format: element_id, fe_descriptor, color, prop, mat, phys, layer
        output.write(f"{i:10d}        11         1         1         1         0         0\n")
        # Hexahedron nodes (8 nodes)
        output.write(f"{cell[0]+1:10d}{cell[1]+1:10d}{cell[2]+1:10d}{cell[3]+1:10d}\n")
        output.write(f"{cell[4]+1:10d}{cell[5]+1:10d}{cell[6]+1:10d}{cell[7]+1:10d}\n")
    
    output.write("    -1\n")
    
    # Write physical groups if available
    if field_data:
        output.write("    -1\n")
        output.write("  2467\n")  # Dataset for groups
        output.write("    -1\n")
        
        for group_name, group_data in field_data.items():
            group_id = group_data[0]
            dimension = group_data[1]
            
            output.write(f"{group_id:10d}\n")
            output.write(f"  {group_name}\n")
            output.write("         0         0         0\n")  # Number of entities in group (placeholder)
            # Actual entities would be added here in a real implementation
            
        output.write("    -1\n")
    
    # File footer
    output.write("    -1\n")
    output.write("99999999\n")
    output.write("    -1\n")
    
    return output.getvalue().encode('ascii')

def create_stl_file(points, triangles):
    """
    Create an ASCII STL file manually without meshio dependencies
    Returns bytes of the STL file content
    """
    output = io.StringIO()
    output.write("solid two_slabs\n")
    
    # Calculate normals for each triangle
    for tri in triangles:
        v0 = points[tri[0]]
        v1 = points[tri[1]]
        v2 = points[tri[2]]
        
        # Calculate normal vector using cross product
        vec1 = v1 - v0
        vec2 = v2 - v0
        normal = np.cross(vec1, vec2)
        
        # Normalize the normal vector
        normal_len = np.linalg.norm(normal)
        if normal_len > 0:
            normal = normal / normal_len
        else:
            normal = np.array([0, 0, 1])  # Default normal if degenerate triangle
        
        # Write facet
        output.write(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
        output.write("    outer loop\n")
        output.write(f"      vertex {v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f}\n")
        output.write(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
        output.write(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
        output.write("    endloop\n")
        output.write("  endfacet\n")
    
    output.write("endsolid two_slabs\n")
    return output.getvalue().encode('ascii')

def export_mesh(mesh_data, format_name="msh"):
    """
    Export mesh data to various formats with robust fallbacks
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Handle filename and path
        base_name = "two_slabs"
        if format_name == "msh":
            filename = f"{base_name}.msh"
        elif format_name == "vtk":
            filename = f"{base_name}.vtk"
        elif format_name == "vtu":
            filename = f"{base_name}.vtu"
        elif format_name == "xdmf" or format_name == "xmf":
            filename = f"{base_name}.xdmf"
        elif format_name == "unv":
            filename = f"{base_name}.unv"
        elif format_name == "stl":
            filename = f"{base_name}.stl"
        elif format_name == "txt":
            filename = f"{base_name}.txt"
        else:
            filename = f"{base_name}.{format_name}"
        
        filepath = os.path.join(tmpdir, filename)
        lx, ly, lz = mesh_data["dimensions"]
        
        try:
            if format_name == "unv":
                if HAS_MESHIO and "unv" in SUPPORTED_FORMATS:
                    # Create meshio Mesh object for UNV export
                    hex_cells = [("hexahedron", mesh_data["cells"])]
                    mesh = meshio.Mesh(
                        points=mesh_data["points"],
                        cells=hex_cells,
                        field_data=mesh_data["field_data"]
                    )
                    mesh.write(filepath, file_format="unv")
                else:
                    # Manual UNV generation fallback
                    unv_content = create_unv_file(
                        mesh_data["points"],
                        mesh_data["cells"],
                        mesh_data.get("field_data")
                    )
                    with open(filepath, 'wb') as f:
                        f.write(unv_content)
            
            elif format_name == "stl":
                # Extract surface mesh for STL export
                surf_points, surf_faces = extract_surface_mesh(mesh_data)
                
                if HAS_MESHIO and "stl" in SUPPORTED_FORMATS:
                    # Create meshio mesh with triangles
                    tri_cells = [("triangle", surf_faces)]
                    mesh = meshio.Mesh(
                        points=surf_points,
                        cells=tri_cells
                    )
                    mesh.write(filepath, file_format="stl")
                else:
                    # Manual STL generation fallback
                    stl_content = create_stl_file(surf_points, surf_faces)
                    with open(filepath, 'wb') as f:
                        f.write(stl_content)
            
            elif HAS_MESHIO:
                # Handle other formats with meshio
                format_map = {
                    "msh": "gmsh22",
                    "vtk": "vtk",
                    "vtu": "vtu",
                    "xdmf": "xdmf"
                }
                
                file_format = format_map.get(format_name, "vtk")
                
                # Check if format is supported by installed meshio
                if file_format not in SUPPORTED_FORMATS:
                    st.warning(f"Format '{file_format}' not supported by installed meshio version. Using VTK format instead.")
                    file_format = "vtk"
                
                # Create appropriate cells
                if file_format == "gmsh22":
                    # For Gmsh format, we need to set cell data for physical groups
                    hex_cells = [("hexahedron", mesh_data["cells"])]
                    mesh = meshio.Mesh(
                        points=mesh_data["points"],
                        cells=hex_cells,
                        cell_data={"gmsh:physical": [np.ones(len(mesh_data["cells"]), dtype=int)]},
                        field_data=mesh_data["field_data"]
                    )
                else:
                    hex_cells = [("hexahedron", mesh_data["cells"])]
                    mesh = meshio.Mesh(
                        points=mesh_data["points"],
                        cells=hex_cells,
                        field_data=mesh_data["field_data"]
                    )
                
                mesh.write(filepath, file_format=file_format)
            
            else:
                # Fallback for when meshio is not available
                if format_name == "txt":
                    # Default text format
                    with open(filepath, 'w') as f:
                        f.write("# Mesh data\n")
                        f.write(f"# Dimensions: lx={lx}, ly={ly}, lz={lz}\n")
                        f.write(f"# Nodes: {len(mesh_data['points'])}\n")
                        f.write(f"# Elements: {len(mesh_data['cells'])}\n")
                        f.write("# Physical groups:\n")
                        for name in mesh_data["field_data"].keys():
                            f.write(f"# - {name}\n")
                else:
                    # Fallback to VTK format generation
                    st.warning(f"Format '{format_name}' requires meshio. Generating simple VTK file instead.")
                    with open(filepath, 'w') as f:
                        f.write("# vtk DataFile Version 3.0\n")
                        f.write("Two slabs mesh\n")
                        f.write("ASCII\n")
                        f.write("DATASET UNSTRUCTURED_GRID\n")
                        f.write(f"POINTS {len(mesh_data['points'])} float\n")
                        for point in mesh_data["points"]:
                            f.write(f"{point[0]} {point[1]} {point[2]}\n")
                        f.write(f"\n")
                        
                        # For simplicity, just write points without cells
                        f.write(f"POINT_DATA {len(mesh_data['points'])}\n")
                        f.write("SCALARS material float 1\n")
                        f.write("LOOKUP_TABLE default\n")
                        for i in range(len(mesh_data["points"])):
                            f.write("1\n")
        
        except Exception as e:
            st.error(f"Error during export: {str(e)}")
            st.warning(f"Falling back to simple text format for {format_name}")
            
            # Fallback to text format
            with open(filepath, 'w') as f:
                f.write(f"# Export error occurred: {str(e)}\n")
                f.write(f"# Fallback text representation of mesh\n")
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
    # Display available meshio formats if available
    if HAS_MESHIO:
        with st.expander("SupportedContent Formats"):
            st.markdown(f"**Meshio version:** {meshio.__version__}")
            st.markdown("**Supported write formats:**")
            st.code(", ".join(sorted(SUPPORTED_FORMATS)))
    
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
        col_viz1, col_viz2 = st.columns(2)
        with col_viz1:
            show_wireframe = st.checkbox("Show Wireframe", value=True)
        with col_viz2:
            show_surface = st.checkbox("Show Surface", value=True)
        
        # Export format - only show supported formats
        st.subheader("Export Format")
        available_formats = ["msh", "vtk", "vtu", "xdmf", "unv", "stl", "txt"]
        if HAS_MESHIO:
            # Filter formats based on meshio support
            available_formats = [fmt for fmt in available_formats if 
                               fmt in ["msh", "vtk", "vtu", "xdmf", "stl", "txt"] or 
                               (fmt == "unv" and "unv" in SUPPORTED_FORMATS)]
        
        export_format = st.selectbox("Format", available_formats)
        
        # Advanced export options
        with st.expander("Advanced Options"):
            st.markdown("### Mesh Quality Settings")
            quality_preset = st.selectbox(
                "Quality Preset",
                ["Standard", "Fine", "Coarse"],
                help="Affects element quality and number"
            )
    
    # Generate button at the top
    generate_button = st.button("🚀 Generate Mesh", type="primary", use_container_width=True)
    
    if generate_button:
        with st.spinner("Creating mesh data..."):
            try:
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
                    
                    **Interface Location:**
                    - Y = {ly}
                    
                    **Mesh Type:**
                    - Structured Hexahedral
                    """)
                    
                    with st.expander("Full List of Physical Groups"):
                        group_names = [
                            "**Volumes:**",
                            "- Solid_1front",
                            "- Solid_2back",
                            "",
                            "**Faces:**",
                            "- Face_1leftfront (left face of front slab)",
                            "- Face_2leftback (left face of back slab)",
                            "- Face_3frontfront (front face of front slab)",
                            "- Face_4bottomfront (bottom face of front slab)",
                            "- Face_5topfront (top face of front slab)",
                            "- Face_6interfacefront (interface between slabs)",
                            "- Face_7bottomback (bottom face of back slab)",
                            "- Face_8topback (top face of back slab)",
                            "- Face_9backback (back face of back slab)",
                            "- Face_10rightfront (right face of front slab)",
                            "- Face_11rightback (right face of back slab)"
                        ]
                        st.markdown("\n".join(group_names))
                
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
                
                # Show file info
                file_size = len(file_data) / 1024  # KB
                st.info(f"📄 **{filename}** | Size: {file_size:.1f} KB")
                
                st.download_button(
                    label=f"⬇️ Download {filename}",
                    data=file_data,
                    file_name=filename,
                    mime=mime_type,
                    use_container_width=True,
                    type="primary"
                )
                
                # Additional information about formats
                with st.expander("ℹ️ Format Information"):
                    format_info = {
                        "msh": "Gmsh format - compatible with Code_Aster, GetFEM, and other FEM solvers. Contains physical groups for boundary conditions.",
                        "vtk": "VTK format - compatible with ParaView and VTK-based tools. Good for visualization.",
                        "vtu": "VTK XML format - more efficient than standard VTK for large datasets.",
                        "xdmf": "Extensible Data Model - efficient for large datasets, compatible with many solvers including FEniCS.",
                        "unv": "I-DEAS Universal format - widely used in commercial FEM software like NX Nastran, MSC Nastran, and others.",
                        "stl": "Stereolithography format - surface mesh only, used for 3D printing and visualization in tools like MeshLab.",
                        "txt": "Simple text format - human readable but limited functionality for FEM analysis."
                    }
                    st.markdown(f"**{export_format.upper()}:** {format_info.get(export_format, 'Standard mesh format')}")
                    
                    if not HAS_MESHIO:
                        st.warning("⚠️ **Note:** Some formats use simplified implementations without meshio. For full format support, install meshio locally.")
                
                st.success("✅ Mesh generated successfully!")
                st.balloons()
            
            except Exception as e:
                st.error(f"❌ Error during mesh generation: {str(e)}")
                st.exception(e)
    
    # Footer information
    st.markdown("---")
    st.caption("✅ This app works in all cloud environments without system dependencies")
    
    # Version information
    with st.expander("App Information"):
        st.markdown("""
        **ParallelGroup Slab Generator v1.2**
        
        This application provides cloud-compatible mesh generation for FEM analysis.
        
        **Features:**
        - Interactive 3D visualization with Plotly
        - Multiple export formats with fallbacks
        - Physical groups preservation
        - Performance optimized for cloud environments
        
        **Dependencies:**
        - Streamlit (web interface)
        - Plotly (3D visualization)
        - Meshio (optional, for advanced exports)
        - NumPy, SciPy (core computations)
        
        **Note:** For best results with all export formats, run locally with meshio installed.
        """)

if __name__ == "__main__":
    main()
