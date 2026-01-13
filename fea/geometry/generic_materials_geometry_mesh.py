# app.py
import streamlit as st
import numpy as np
import tempfile
import os
import plotly.graph_objects as go
import io

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
MESHIO_VERSION = "Not installed"
try:
    import meshio
    HAS_MESHIO = True
    MESHIO_VERSION = meshio.__version__
    st.success(f"✅ Meshio {MESHIO_VERSION} available")
except Exception as e:
    st.warning(f"⚠️ Meshio not available: {str(e)}. Will use simplified exports.")

# Define standard export formats (don't rely on internal meshio attributes)
STANDARD_FORMATS = ["msh", "vtk", "vtu", "xdmf", "unv", "stl", "txt"]

def create_hex_mesh(lx, ly, lz, resolution=10):
    """Create a hexahedral mesh without external dependencies"""
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
    """Extract surface faces from hexahedral mesh for STL export"""
    points = mesh_data["points"]
    cells = mesh_data["cells"]
    
    # Dictionary to track face usage
    face_dict = {}
    
    for cell_idx, hex_nodes in enumerate(cells):
        # Define the 6 faces of a hexahedron
        hex_faces = [
            [hex_nodes[0], hex_nodes[1], hex_nodes[2], hex_nodes[3]],  # bottom
            [hex_nodes[4], hex_nodes[5], hex_nodes[6], hex_nodes[7]],  # top
            [hex_nodes[0], hex_nodes[1], hex_nodes[5], hex_nodes[4]],  # front
            [hex_nodes[2], hex_nodes[3], hex_nodes[7], hex_nodes[6]],  # back
            [hex_nodes[0], hex_nodes[3], hex_nodes[7], hex_nodes[4]],  # left
            [hex_nodes[1], hex_nodes[2], hex_nodes[6], hex_nodes[5]]   # right
        ]
        
        for face_nodes in hex_faces:
            # Create canonical key for the face
            face_key = tuple(sorted(face_nodes))
            
            if face_key not in face_dict:
                face_dict[face_key] = 0
            face_dict[face_key] += 1
    
    # Surface faces are those used by only one cell
    surface_quads = [face for face, count in face_dict.items() if count == 1]
    
    # Convert quadrilateral faces to triangles for STL
    tri_faces = []
    for quad in surface_quads:
        tri_faces.append([quad[0], quad[1], quad[2]])
        tri_faces.append([quad[0], quad[2], quad[3]])
    
    return points, np.array(tri_faces)

def visualize_with_plotly(mesh_data, show_wireframe=True, show_surface=True):
    """Create interactive 3D visualization using Plotly"""
    points = mesh_data["points"]
    cells = mesh_data["cells"]
    lx, ly, lz = mesh_data["dimensions"]
    
    fig = go.Figure()
    
    # Add wireframe visualization (limit for performance)
    if show_wireframe and len(cells) > 0:
        max_cells = min(100, len(cells))  # Limit for cloud performance
        for hex_nodes in cells[:max_cells]:
            edges = [
                [hex_nodes[0], hex_nodes[1]], [hex_nodes[1], hex_nodes[2]], 
                [hex_nodes[2], hex_nodes[3]], [hex_nodes[3], hex_nodes[0]],
                [hex_nodes[4], hex_nodes[5]], [hex_nodes[5], hex_nodes[6]], 
                [hex_nodes[6], hex_nodes[7]], [hex_nodes[7], hex_nodes[4]],
                [hex_nodes[0], hex_nodes[4]], [hex_nodes[1], hex_nodes[5]], 
                [hex_nodes[2], hex_nodes[6]], [hex_nodes[3], hex_nodes[7]]
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
    
    # Highlight the interface between the two slabs
    interface_y = ly
    fig.add_trace(go.Surface(
        x=[[0, lx], [0, lx]],
        y=[[interface_y, interface_y], [interface_y, interface_y]],
        z=[[0, 0], [lz, lz]],
        colorscale=[[0, 'red'], [1, 'red']],
        opacity=0.7,
        name='Interface (Face_6interfacefront)',
        showscale=False
    ))
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        title='3D Mesh Visualization',
        height=600,
        margin=dict(l=0, r=0, b=0, t=30),
        showlegend=True
    )
    
    return fig

def create_unv_content(points, cells):
    """Create UNV file content manually (without meshio)"""
    output = io.StringIO()
    
    # Write header
    output.write("    -1\n")
    output.write("  2411\n")  # Node dataset
    output.write("    -1\n")
    output.write("         1\n")  # Dataset ID
    output.write(f"{len(points):10d}\n")  # Number of nodes
    
    # Write nodes
    for i, point in enumerate(points, 1):
        output.write(f"{i:10d}         1         1         0\n")
        output.write(f"{point[0]:13.5E}{point[1]:13.5E}{point[2]:13.5E}\n")
    
    output.write("    -1\n")
    output.write("  2412\n")  # Element dataset
    output.write("    -1\n")
    output.write(f"{len(cells):10d}\n")  # Number of elements
    
    # Write elements (hexahedrons)
    for i, cell in enumerate(cells, 1):
        output.write(f"{i:10d}        11         1         1         1         0         0\n")
        output.write(f"{cell[0]+1:10d}{cell[1]+1:10d}{cell[2]+1:10d}{cell[3]+1:10d}\n")
        output.write(f"{cell[4]+1:10d}{cell[5]+1:10d}{cell[6]+1:10d}{cell[7]+1:10d}\n")
    
    output.write("    -1\n")
    output.write("99999999\n")
    output.write("    -1\n")
    
    return output.getvalue().encode('ascii')

def create_stl_content(points, triangles):
    """Create STL file content manually (without meshio)"""
    output = io.StringIO()
    output.write("solid two_slabs\n")
    
    for tri in triangles:
        v0 = points[tri[0]]
        v1 = points[tri[1]]
        v2 = points[tri[2]]
        
        # Calculate normal using cross product
        vec1 = v1 - v0
        vec2 = v2 - v0
        normal = np.cross(vec1, vec2)
        
        # Normalize
        norm_len = np.linalg.norm(normal)
        if norm_len > 0:
            normal = normal / norm_len
        else:
            normal = np.array([0, 0, 1])
        
        output.write(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
        output.write("    outer loop\n")
        output.write(f"      vertex {v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f}\n")
        output.write(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
        output.write(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
        output.write("    endloop\n")
        output.write("  endfacet\n")
    
    output.write("endsolid two_slabs\n")
    return output.getvalue().encode('ascii')

def export_mesh(mesh_data, format_name="vtk"):
    """Export mesh with robust fallbacks for all environments"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Determine filename
        filename = f"two_slabs.{format_name}"
        filepath = os.path.join(tmpdir, filename)
        lx, ly, lz = mesh_data["dimensions"]
        
        try:
            if format_name == "unv":
                if HAS_MESHIO:
                    # Try using meshio first
                    try:
                        hex_cells = [("hexahedron", mesh_data["cells"])]
                        mesh = meshio.Mesh(
                            points=mesh_data["points"],
                            cells=hex_cells,
                            field_data=mesh_data["field_data"]
                        )
                        mesh.write(filepath, file_format="unv")
                    except Exception as e1:
                        st.warning(f"Meshio UNV export failed: {str(e1)}. Using manual export.")
                        # Fallback to manual UNV generation
                        unv_content = create_unv_content(
                            mesh_data["points"],
                            mesh_data["cells"]
                        )
                        with open(filepath, 'wb') as f:
                            f.write(unv_content)
                else:
                    # Manual UNV generation
                    unv_content = create_unv_content(
                        mesh_data["points"],
                        mesh_data["cells"]
                    )
                    with open(filepath, 'wb') as f:
                        f.write(unv_content)
            
            elif format_name == "stl":
                # Extract surface for STL
                surf_points, surf_faces = extract_surface_mesh(mesh_data)
                
                if HAS_MESHIO:
                    # Try using meshio first
                    try:
                        tri_cells = [("triangle", surf_faces)]
                        mesh = meshio.Mesh(
                            points=surf_points,
                            cells=tri_cells
                        )
                        mesh.write(filepath, file_format="stl")
                    except Exception as e1:
                        st.warning(f"Meshio STL export failed: {str(e1)}. Using manual export.")
                        # Fallback to manual STL generation
                        stl_content = create_stl_content(surf_points, surf_faces)
                        with open(filepath, 'wb') as f:
                            f.write(stl_content)
                else:
                    # Manual STL generation
                    stl_content = create_stl_content(surf_points, surf_faces)
                    with open(filepath, 'wb') as f:
                        f.write(stl_content)
            
            elif HAS_MESHIO and format_name in ["msh", "vtk", "vtu", "xdmf"]:
                # Handle other meshio-supported formats
                hex_cells = [("hexahedron", mesh_data["cells"])]
                mesh = meshio.Mesh(
                    points=mesh_data["points"],
                    cells=hex_cells,
                    field_data=mesh_data["field_data"]
                )
                
                # Map format names to meshio internal names
                format_map = {
                    "msh": "gmsh22",
                    "vtk": "vtk",
                    "vtu": "vtu",
                    "xdmf": "xdmf"
                }
                
                file_format = format_map.get(format_name, "vtk")
                mesh.write(filepath, file_format=file_format)
            
            else:
                # Fallback text format
                with open(filepath, 'w') as f:
                    f.write("# Mesh data\n")
                    f.write(f"# Dimensions: lx={lx}, ly={ly}, lz={lz}\n")
                    f.write(f"# Nodes: {len(mesh_data['points'])}\n")
                    f.write(f"# Elements: {len(mesh_data['cells'])}\n")
                    f.write("# Physical groups:\n")
                    for name in mesh_data["field_data"].keys():
                        f.write(f"# - {name}\n")
        
        except Exception as e:
            st.error(f"Export error: {str(e)}. Generating fallback text format.")
            # Ultimate fallback - simple text format
            with open(filepath, 'w') as f:
                f.write(f"# Export failed for {format_name} format\n")
                f.write(f"# Error: {str(e)}\n")
                f.write(f"# Dimensions: lx={lx}, ly={ly}, lz={lz}\n")
                f.write(f"# Nodes: {len(mesh_data['points'])}\n")
                f.write(f"# Elements: {len(mesh_data['cells'])}\n")
        
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
        
        # Export format
        st.subheader("Export Format")
        export_format = st.selectbox("Format", STANDARD_FORMATS)
    
    # Generate button
    if st.button("🚀 Generate Mesh", type="primary", use_container_width=True):
        with st.spinner("Creating mesh data..."):
            try:
                # Create the mesh
                mesh_data = create_hex_mesh(lx, ly, lz, resolution)
                
                # Get statistics
                num_points = len(mesh_data["points"])
                num_cells = len(mesh_data["cells"])
                
                # Display statistics
                st.subheader("📊 Mesh Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Nodes", f"{num_points:,}")
                with col2:
                    st.metric("Elements", f"{num_cells:,}")
                with col3:
                    st.metric("Physical Groups", "13")
                
                # Visualization and info side by side
                col_viz, col_info = st.columns([2, 1])
                
                # 3D Visualization
                with col_viz:
                    st.subheader("3D Visualization")
                    fig = visualize_with_plotly(mesh_data, show_wireframe)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Physical groups information
                with col_info:
                    st.subheader("PropertyParams")
                    st.markdown(f"""
                    **Dimensions:**
                    - Length (X): {lx}
                    - Width (Y): {2*ly} (total)
                    - Height (Z): {lz}
                    
                    **Mesh Type:** Hexahedral
                    **Interface:** Y = {ly}
                    """)
                    
                    with st.expander("Physical Groups"):
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
                        - Face_6interfacefront
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
                    "vtk": "application/octet-stream",
                    "vtu": "application/octet-stream",
                    "xdmf": "application/octet-stream",
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
                
                st.success("✅ Mesh generated successfully!")
            
            except Exception as e:
                st.error(f"❌ Error during mesh generation: {str(e)}")
                st.exception(e)
    
    # Footer
    st.markdown("---")
    st.caption("✅ This app works in all cloud environments without system dependencies")

if __name__ == "__main__":
    main()
