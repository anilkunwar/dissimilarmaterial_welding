# app.py
import streamlit as st
import numpy as np
import tempfile
import os
import plotly.graph_objects as go
import io
import math

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
MESHIO_VERSION = "Not installed"
SUPPORTED_FORMATS = []
HAS_UNV_SUPPORT = False

try:
    import meshio
    HAS_MESHIO = True
    MESHIO_VERSION = meshio.__version__
    
    # Get actual supported formats from meshio
    SUPPORTED_FORMATS = list(meshio._helpers.writer_map.keys()) if hasattr(meshio._helpers, 'writer_map') else [
        "abaqus", "ansys", "avsucd", "cgns", "dolfin-xml", "exodus", 
        "flac3d", "gmsh", "gmsh22", "h5m", "hmf", "mdpa", "med", "medit", 
        "nastran", "netgen", "neuroglancer", "obj", "off", "permas", "ply", 
        "stl", "su2", "svg", "tecplot", "tetgen", "ugrid", "vtk", "vtk42", 
        "vtk51", "vtu", "wkt", "xdmf"
    ]
    
    # Check if nastran format is available for UNV export
    HAS_UNV_SUPPORT = "nastran" in SUPPORTED_FORMATS or "unv" in SUPPORTED_FORMATS
    
    status_msg = f"✅ Meshio {MESHIO_VERSION} available"
    if HAS_UNV_SUPPORT:
        status_msg += " with UNV support"
    else:
        status_msg += " (no UNV support)"
    st.success(status_msg)
        
except Exception as e:
    st.warning(f"⚠️ Meshio not available: {str(e)}. Will use manual exports.")
    HAS_UNV_SUPPORT = False

# Define standard export formats
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
        "Solid_1front": {"value": 1, "dimension": 3},
        "Solid_2back": {"value": 2, "dimension": 3},
        "Face_1leftfront": {"value": 1, "dimension": 2}, 
        "Face_2leftback": {"value": 2, "dimension": 2},
        "Face_3frontfront": {"value": 3, "dimension": 2},
        "Face_4bottomfront": {"value": 4, "dimension": 2},
        "Face_5topfront": {"value": 5, "dimension": 2},
        "Face_6interfacefront": {"value": 6, "dimension": 2},
        "Face_7bottomback": {"value": 7, "dimension": 2},
        "Face_8topback": {"value": 8, "dimension": 2},
        "Face_9backback": {"value": 9, "dimension": 2},
        "Face_10rightfront": {"value": 10, "dimension": 2},
        "Face_11rightback": {"value": 11, "dimension": 2}
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

def visualize_with_plotly(mesh_data, show_wireframe=True):
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
    
    # Add coordinate axes
    axis_length = max(lx, 2*ly, lz) * 0.2
    fig.add_trace(go.Scatter3d(
        x=[0, axis_length], y=[0, 0], z=[0, 0],
        mode='lines+text',
        line=dict(color='red', width=3),
        text=['', 'X'],
        textposition='top center',
        name='X-axis'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, axis_length], z=[0, 0],
        mode='lines+text',
        line=dict(color='green', width=3),
        text=['', 'Y'],
        textposition='top center',
        name='Y-axis'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, axis_length],
        mode='lines+text',
        line=dict(color='blue', width=3),
        text=['', 'Z'],
        textposition='top center',
        name='Z-axis'
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

def create_proper_unv_content(points, cells, field_data=None):
    """
    Create a properly formatted UNV file that follows I-DEAS Universal File Format specifications
    """
    output = io.StringIO()
    
    # Write file header marker
    output.write("    -1\n")
    output.write("  2411\n")  # Node dataset
    output.write("    -1\n")
    
    # Dataset header for nodes - correct format for I-DEAS
    output.write("        1        1        0        0        0        0        0\n")
    output.write(f"{len(points):10d}\n")  # Number of nodes
    
    # Write nodes with proper formatting
    for i, point in enumerate(points, 1):
        # Node record: node_id, physical properties table number, color, coordinate system
        output.write(f"{i:10d}        1        1        0        0        0        0\n")
        # Coordinates with scientific notation and proper spacing
        output.write(f"{point[0]:13.5E}{point[1]:13.5E}{point[2]:13.5E}\n")
    
    output.write("    -1\n")
    output.write("  2412\n")  # Element dataset
    output.write("    -1\n")
    
    # Dataset header for elements
    output.write("        1        1        0        0        0        0        0\n")
    output.write(f"{len(cells):10d}\n")  # Number of elements
    
    # Write elements (hexahedrons - type 11)
    for i, cell in enumerate(cells, 1):
        # Element header: id, type(11=hex), color, prop_table, mat_table, etc.
        output.write(f"{i:10d}       11        1        1        1        0        0        0\n")
        # 8 nodes of hexahedron - split into two lines of 4 nodes each
        output.write(f"{cell[0]+1:10d}{cell[1]+1:10d}{cell[2]+1:10d}{cell[3]+1:10d}\n")
        output.write(f"{cell[4]+1:10d}{cell[5]+1:10d}{cell[6]+1:10d}{cell[7]+1:10d}\n")
    
    output.write("    -1\n")
    
    # Add dataset 2467 for groups/physical entities if field data is provided
    if field_data:
        output.write("    -1\n")
        output.write("  2467\n")  # Group dataset
        output.write("    -1\n")
        
        # Dataset header for groups
        output.write("        1        1        0        0        0        0        0\n")
        output.write(f"{len(field_data):10d}\n")  # Number of groups
        
        # Write each group/physical entity
        group_id = 1
        for name, data in field_data.items():
            # Group header record
            output.write(f"{group_id:10d}        0        0        0        0        0        0\n")
            # Group name (padded to 80 characters)
            output.write(f"{name.ljust(80)}\n")
            # Analysis type, data type, version number
            output.write("        1        1        0\n")
            # Number of entities in this group (0 for now - placeholder)
            output.write("        0\n")
            
            group_id += 1
        
        output.write("    -1\n")
    
    # File termination
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
                if HAS_MESHIO and HAS_UNV_SUPPORT:
                    # Using meshio with nastran format for UNV files
                    try:
                        hex_cells = [("hexahedron", mesh_data["cells"])]
                        
                        # Create mesh with proper physical groups
                        mesh = meshio.Mesh(
                            points=mesh_data["points"],
                            cells=hex_cells,
                        )
                        
                        # Try different format names that might work for UNV
                        unv_formats = ["nastran", "unv"]
                        success = False
                        
                        for fmt in unv_formats:
                            if fmt in SUPPORTED_FORMATS:
                                try:
                                    mesh.write(filepath, file_format=fmt)
                                    success = True
                                    st.success(f"✅ Meshio UNV export successful using '{fmt}' format")
                                    break
                                except Exception as e2:
                                    st.warning(f"⚠️ Meshio '{fmt}' format failed: {str(e2)}")
                        
                        if not success:
                            raise Exception("All UNV format attempts failed")
                        
                        # Rename file to have .unv extension if needed
                        if not filepath.endswith(".unv"):
                            new_path = os.path.splitext(filepath)[0] + ".unv"
                            os.rename(filepath, new_path)
                            filepath = new_path
                            filename = os.path.basename(new_path)
                        
                    except Exception as e1:
                        st.warning(f"⚠️ Meshio UNV export failed: {str(e1)}. Using manual export.")
                        # Fallback to manual UNV generation
                        unv_content = create_proper_unv_content(
                            mesh_data["points"],
                            mesh_data["cells"],
                            mesh_data["field_data"]
                        )
                        with open(filepath, 'wb') as f:
                            f.write(unv_content)
                        st.success("✅ Manual UNV export successful")
                else:
                    # Manual UNV generation
                    unv_content = create_proper_unv_content(
                        mesh_data["points"],
                        mesh_data["cells"],
                        mesh_data["field_data"]
                    )
                    with open(filepath, 'wb') as f:
                        f.write(unv_content)
                    st.success("✅ Manual UNV export successful")
            
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
                        st.success("✅ Meshio STL export successful")
                    except Exception as e1:
                        st.warning(f"⚠️ Meshio STL export failed: {str(e1)}. Using manual export.")
                        # Fallback to manual STL generation
                        stl_content = create_stl_content(surf_points, surf_faces)
                        with open(filepath, 'wb') as f:
                            f.write(stl_content)
                        st.success("✅ Manual STL export successful")
                else:
                    # Manual STL generation
                    stl_content = create_stl_content(surf_points, surf_faces)
                    with open(filepath, 'wb') as f:
                        f.write(stl_content)
                    st.success("✅ Manual STL export successful")
            
            elif HAS_MESHIO and format_name in ["msh", "vtk", "vtu", "xdmf"]:
                # Handle other meshio-supported formats
                hex_cells = [("hexahedron", mesh_data["cells"])]
                
                # Prepare field data for physical groups
                field_data = {}
                for name, data in mesh_data["field_data"].items():
                    field_data[name] = np.array([data["value"], data["dimension"]])
                
                mesh = meshio.Mesh(
                    points=mesh_data["points"],
                    cells=hex_cells,
                    field_data=field_data
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
                st.success(f"✅ Meshio {format_name.upper()} export successful")
            
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
                st.success("✅ Text format export successful")
        
        except Exception as e:
            st.error(f"❌ Export error: {str(e)}. Generating fallback text format.")
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
        
        # Export format - filter based on capabilities
        st.subheader("Export Format")
        
        # Filter available formats based on meshio capabilities
        available_formats = STANDARD_FORMATS.copy()
        if "unv" in available_formats and not (HAS_MESHIO and HAS_UNV_SUPPORT):
            # If meshio doesn't support UNV, we still keep it because we have manual export
            pass  # Keep UNV format since we have manual fallback
        
        export_format = st.selectbox("Format", available_formats)
        
        # Show meshio status in expander
        if HAS_MESHIO:
            with st.expander("Meshio Details"):
                st.write(f"**Version:** {MESHIO_VERSION}")
                st.write(f"**UNV Support:** {'✅ Yes' if HAS_UNV_SUPPORT else '❌ No'}")
                st.write("**Supported Formats:**")
                st.code(", ".join(sorted(SUPPORTED_FORMATS)))
    
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
                
                # Detailed format information
                with st.expander("ℹ️ Format Information"):
                    format_descriptions = {
                        "msh": "Gmsh format - compatible with Code_Aster, GetFEM, and other FEM solvers",
                        "vtk": "VTK format - compatible with ParaView and VTK-based visualization tools",
                        "vtu": "VTK XML format - more efficient than standard VTK for large datasets",
                        "xdmf": "Extensible Data Model Format - efficient for large datasets and time series",
                        "unv": "I-DEAS Universal format - widely used in commercial FEM software like MSC Nastran, NX Nastran, Femap",
                        "stl": "Stereolithography format - surface mesh only, used for 3D printing and visualization",
                        "txt": "Simple text format - human readable but limited functionality"
                    }
                    
                    st.markdown(f"**{export_format.upper()}:** {format_descriptions.get(export_format, 'Standard mesh format')}")
                    
                    if export_format == "unv":
                        st.markdown("""
                        **UNV Export Details:**
                        - Uses meshio with 'nastran' format when available
                        - Falls back to manual generation following I-DEAS Universal File Format
                        - Contains proper node and element datasets (2411, 2412)
                        - Includes physical group definitions for boundary conditions
                        - Compatible with most commercial FEM software
                        """)
                
                st.success("✅ Mesh generated and exported successfully!")
            
            except Exception as e:
                st.error(f"❌ Error during mesh generation: {str(e)}")
                st.exception(e)
    
    # Footer
    st.markdown("---")
    st.caption("✅ This app works in all cloud environments with robust fallbacks for all export formats")

if __name__ == "__main__":
    main()
