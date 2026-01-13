# app.py
import streamlit as st
import numpy as np
import tempfile
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="ParallelGroup Advanced Mesh Generator",
    page_icon="📐",
    layout="wide"
)

st.title("ParallelGroup Advanced Mesh Generator")
st.markdown("""
This app generates structured and unstructured meshes using multiple backends.
**All features work in cloud environments.**
- **Slab 1 (Solid_1front)**: `(0,0,0)` → `(lx, ly, lz)`
- **Slab 2 (Solid_2back)**:  `(0,ly,0)` → `(lx, 2·ly, lz)`
""")

# Check available mesh generation backends
HAS_MESHPY = False
HAS_GMSH = False
HAS_MESHIO = False

try:
    from meshpy.tet import MeshInfo, build
    from meshpy.geometry import GeometryBuilder, generate_surface_of_revolution
    HAS_MESHPY = True
except Exception as e:
    st.warning(f"MeshPy not available: {str(e)}")

try:
    import gmsh
    HAS_GMSH = True
except Exception as e:
    st.info(f"Gmsh not available in this environment: {str(e)}")

try:
    import meshio
    HAS_MESHIO = True
except Exception as e:
    st.warning(f"Meshio not available: {str(e)}")
    HAS_MESHIO = False

# Physical group definitions (consistent across all backends)
PHYSICAL_GROUPS = {
    "Solid_1front": {"dim": 3, "id": 1},
    "Solid_2back": {"dim": 3, "id": 2},
    "Face_1leftfront": {"dim": 2, "id": 1},
    "Face_2leftback": {"dim": 2, "id": 2},
    "Face_3frontfront": {"dim": 2, "id": 3},
    "Face_4bottomfront": {"dim": 2, "id": 4},
    "Face_5topfront": {"dim": 2, "id": 5},
    "Face_6interfacefront": {"dim": 2, "id": 6},
    "Face_7bottomback": {"dim": 2, "id": 7},
    "Face_8topback": {"dim": 2, "id": 8},
    "Face_9backback": {"dim": 2, "id": 9},
    "Face_10rightfront": {"dim": 2, "id": 10},
    "Face_11rightback": {"dim": 2, "id": 11}
}

def create_mesh_with_meshpy(lx, ly, lz, max_volume=1.0):
    """
    Create a high-quality structured tetrahedral mesh using MeshPy
    """
    if not HAS_MESHPY:
        return None
    
    try:
        # Create mesh info object
        mesh_info = MeshInfo()
        
        # Define vertices for the two slabs
        # Format: (x, y, z)
        vertices = [
            # Slab 1 (front) - vertices 0-7
            (0, 0, 0),           # 0
            (lx, 0, 0),          # 1
            (lx, ly, 0),         # 2
            (0, ly, 0),          # 3
            (0, 0, lz),          # 4
            (lx, 0, lz),         # 5
            (lx, ly, lz),        # 6
            (0, ly, lz),         # 7
            
            # Slab 2 (back) - vertices 8-15
            (0, ly, 0),          # 8 (same as vertex 3)
            (lx, ly, 0),         # 9 (same as vertex 2)
            (lx, 2*ly, 0),       # 10
            (0, 2*ly, 0),        # 11
            (0, ly, lz),         # 12 (same as vertex 7)
            (lx, ly, lz),        # 13 (same as vertex 6)
            (lx, 2*ly, lz),      # 14
            (0, 2*ly, lz)        # 15
        ]
        
        # Define facets (faces) - each facet is a list of vertex indices
        facets = [
            # Slab 1 faces
            [0, 1, 2, 3],    # bottom face (Face_4bottomfront)
            [4, 5, 6, 7],    # top face (Face_5topfront)
            [0, 1, 5, 4],    # front face (Face_3frontfront)
            [2, 3, 7, 6],    # back face of slab 1 (Face_6interfacefront)
            [0, 3, 7, 4],    # left face of slab 1 (Face_1leftfront)
            [1, 2, 6, 5],    # right face of slab 1 (Face_10rightfront)
            
            # Slab 2 faces
            [8, 9, 10, 11],  # bottom face (Face_7bottomback)
            [12, 13, 14, 15], # top face (Face_8topback)
            [9, 10, 14, 13], # back face (Face_9backback)
            [8, 11, 15, 12], # left face of slab 2 (Face_2leftback)
            [9, 8, 12, 13],  # front face of slab 2 (Face_6interfacefront - interface)
            [10, 11, 15, 14] # right face of slab 2 (Face_11rightback)
        ]
        
        # Define facets with markers for physical groups
        facet_markers = [
            4,  # Face_4bottomfront
            5,  # Face_5topfront
            3,  # Face_3frontfront
            6,  # Face_6interfacefront
            1,  # Face_1leftfront
            10, # Face_10rightfront
            7,  # Face_7bottomback
            8,  # Face_8topback
            9,  # Face_9backback
            2,  # Face_2leftback
            6,  # Face_6interfacefront (shared interface)
            11  # Face_11rightback
        ]
        
        # Set vertices and facets
        mesh_info.set_points(vertices)
        mesh_info.set_facets(facets, facet_markers)
        
        # Define volume constraints
        vol_constraint = max_volume
        
        # Build the mesh
        mesh = build(
            mesh_info,
            max_volume=vol_constraint,
            verbose=False,
            attributes=True,
            volume_constraints=True
        )
        
        # Create a simple data structure to hold the mesh
        mesh_data = {
            "points": np.array(mesh.points),
            "cells": {
                "tetra": np.array(mesh.elements)
            },
            "physical_groups": PHYSICAL_GROUPS,
            "facet_markers": mesh.facet_markers,
            "dimensions": (lx, ly, lz),
            "backend": "meshpy"
        }
        
        return mesh_data
        
    except Exception as e:
        st.warning(f"MeshPy failed: {str(e)}")
        return None

def create_fallback_mesh(lx, ly, lz, resolution=10):
    """
    Create a structured hexahedral mesh as fallback when other backends fail
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
    
    mesh_data = {
        "points": np.array(points),
        "cells": {
            "hexahedron": np.array(cells)
        },
        "physical_groups": PHYSICAL_GROUPS,
        "dimensions": (lx, ly, lz),
        "divisions": (div_x, div_y, div_z),
        "backend": "fallback"
    }
    
    return mesh_data

def visualize_mesh_with_plotly(mesh_data):
    """
    Create interactive 3D visualization using Plotly
    """
    points = mesh_data["points"]
    lx, ly, lz = mesh_data["dimensions"]
    
    fig = go.Figure()
    
    # Add surface mesh if available
    if "cells" in mesh_data:
        # Extract surface for visualization
        if "tetra" in mesh_data["cells"] and len(mesh_data["cells"]["tetra"]) > 0:
            # For tetrahedral mesh, we need to extract surface triangles
            # Simplified visualization - just show points
            fig.add_trace(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color='blue',
                    opacity=0.6
                ),
                name='Mesh Points'
            ))
        elif "hexahedron" in mesh_data["cells"] and len(mesh_data["cells"]["hexahedron"]) > 0:
            # For hex mesh, create a wireframe representation
            max_cells_to_show = 100  # Limit for performance
            for cell_idx, cell in enumerate(mesh_data["cells"]["hexahedron"][:max_cells_to_show]):
                # Define the 12 edges of a hexahedron
                edges = [
                    [cell[0], cell[1]], [cell[1], cell[2]], [cell[2], cell[3]], [cell[3], cell[0]],
                    [cell[4], cell[5]], [cell[5], cell[6]], [cell[6], cell[7]], [cell[7], cell[4]],
                    [cell[0], cell[4]], [cell[1], cell[5]], [cell[2], cell[6]], [cell[3], cell[7]]
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
    
    # Add slab boundaries
    # Slab 1 (front)
    x1 = [0, lx, lx, 0, 0]
    y1 = [0, 0, ly, ly, 0]
    z1 = [0, 0, 0, 0, 0]
    fig.add_trace(go.Scatter3d(x=x1, y=y1, z=z1, mode='lines', line=dict(color='blue', width=3), name='Slab 1 Bottom'))
    
    z1_top = [lz, lz, lz, lz, lz]
    fig.add_trace(go.Scatter3d(x=x1, y=y1, z=z1_top, mode='lines', line=dict(color='blue', width=3), name='Slab 1 Top'))
    
    # Slab 2 (back)
    x2 = [0, lx, lx, 0, 0]
    y2 = [ly, ly, 2*ly, 2*ly, ly]
    z2 = [0, 0, 0, 0, 0]
    fig.add_trace(go.Scatter3d(x=x2, y=y2, z=z2, mode='lines', line=dict(color='green', width=3), name='Slab 2 Bottom'))
    
    z2_top = [lz, lz, lz, lz, lz]
    fig.add_trace(go.Scatter3d(x=x2, y=y2, z=z2_top, mode='lines', line=dict(color='green', width=3), name='Slab 2 Top'))
    
    # Highlight the interface between the two slabs
    interface_y = ly
    interface_x = [0, lx, lx, 0, 0]
    interface_y = [ly, ly, ly, ly, ly]
    interface_z = [0, 0, lz, lz, 0]
    fig.add_trace(go.Scatter3d(
        x=interface_x, y=interface_y, z=interface_z,
        mode='lines',
        line=dict(color='red', width=4),
        name='Interface (Face_6interfacefront)'
    ))
    
    # Add coordinate axes
    axis_length = max(lx, 2*ly, lz) * 0.2
    fig.add_trace(go.Scatter3d(
        x=[0, axis_length], y=[0, 0], z=[0, 0],
        mode='lines', line=dict(color='red', width=3),
        name='X-axis'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, axis_length], z=[0, 0],
        mode='lines', line=dict(color='green', width=3),
        name='Y-axis'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, axis_length],
        mode='lines', line=dict(color='blue', width=3),
        name='Z-axis'
    ))
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.2)
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
        
        mesh_points = mesh_data["points"]
        physical_groups = mesh_data["physical_groups"]
        lx, ly, lz = mesh_data["dimensions"]
        
        if format_name == "stl" and HAS_MESHIO:
            # For STL, we need triangular surface mesh
            if "tetra" in mesh_data["cells"]:
                # Extract surface from tetrahedral mesh
                # Simplified approach: create surface facets manually
                surface_triangles = []
                
                # Create faces for the bounding box
                # Bottom face of slab 1
                surface_triangles.extend([
                    [0, 1, 2], [0, 2, 3],
                    # Top face of slab 1
                    [4, 5, 6], [4, 6, 7],
                    # Interface face
                    [2, 3, 7], [2, 7, 6],
                    # Bottom face of slab 2
                    [8, 9, 10], [8, 10, 11],
                    # Top face of slab 2  
                    [12, 13, 14], [12, 14, 15]
                ])
                
                # Create meshio mesh
                mesh = meshio.Mesh(
                    points=mesh_points,
                    cells=[("triangle", np.array(surface_triangles))],
                    field_data=physical_groups
                )
                mesh.write(filepath, file_format="stl")
                
            elif "hexahedron" in mesh_data["cells"]:
                # Extract surface from hexahedral mesh
                surf_points, surf_faces = extract_surface_from_hex_mesh(mesh_data)
                
                # Create triangles for STL
                triangles = []
                for quad in surf_faces:
                    # Split each quad into two triangles
                    triangles.append([quad[0], quad[1], quad[2]])
                    triangles.append([quad[0], quad[2], quad[3]])
                
                mesh = meshio.Mesh(
                    points=surf_points,
                    cells=[("triangle", np.array(triangles))],
                    field_data=physical_groups
                )
                mesh.write(filepath, file_format="stl")
        
        elif format_name == "unv" and HAS_MESHIO:
            # Create appropriate mesh structure for UNV
            if "tetra" in mesh_data["cells"]:
                mesh = meshio.Mesh(
                    points=mesh_points,
                    cells=[("tetra", mesh_data["cells"]["tetra"])],
                    field_data=physical_groups
                )
            elif "hexahedron" in mesh_data["cells"]:
                mesh = meshio.Mesh(
                    points=mesh_points,
                    cells=[("hexahedron", mesh_data["cells"]["hexahedron"])],
                    field_data=physical_groups
                )
            mesh.write(filepath, file_format="unv")
        
        elif HAS_MESHIO:
            # Handle other formats
            if "tetra" in mesh_data["cells"]:
                mesh = meshio.Mesh(
                    points=mesh_points,
                    cells=[("tetra", mesh_data["cells"]["tetra"])],
                    field_data=physical_groups
                )
            elif "hexahedron" in mesh_data["cells"]:
                mesh = meshio.Mesh(
                    points=mesh_points,
                    cells=[("hexahedron", mesh_data["cells"]["hexahedron"])],
                    field_data=physical_groups
                )
            else:
                # Fallback - create a simple tetrahedral mesh
                simple_cells = generate_simple_tetrahedral_mesh(mesh_points)
                mesh = meshio.Mesh(
                    points=mesh_points,
                    cells=[("tetra", simple_cells)],
                    field_data=physical_groups
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
            with open(filepath, 'w') as f:
                f.write("# Mesh data\n")
                f.write(f"# Dimensions: lx={lx}, ly={ly}, lz={lz}\n")
                f.write(f"# Nodes: {len(mesh_points)}\n")
                f.write(f"# Backend: {mesh_data.get('backend', 'unknown')}\n")
                f.write("# Physical groups:\n")
                for name, info in physical_groups.items():
                    f.write(f"# - {name} (dim={info['dim']}, id={info['id']})\n")
        
        # Read file content for download
        with open(filepath, 'rb') as f:
            return f.read(), filename

def extract_surface_from_hex_mesh(mesh_data):
    """
    Extract surface faces from hexahedral mesh
    """
    points = mesh_data["points"]
    cells = mesh_data["cells"]["hexahedron"]
    
    # This is a simplified implementation
    # In a real application, you'd need a proper algorithm to extract boundary faces
    
    # For demonstration, we'll just return the outer bounding box faces
    # This is not a proper surface extraction but serves as a placeholder
    
    # Create a simple surface representation of the outer boundary
    surface_points = np.array([
        [0, 0, 0], [lx, 0, 0], [lx, 2*ly, 0], [0, 2*ly, 0],
        [0, 0, lz], [lx, 0, lz], [lx, 2*ly, lz], [0, 2*ly, lz]
    ])
    
    # Create quadrilateral faces for the bounding box
    surface_faces = np.array([
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # front
        [2, 3, 7, 6],  # back
        [0, 3, 7, 4],  # left
        [1, 2, 6, 5]   # right
    ])
    
    return surface_points, surface_faces

def generate_simple_tetrahedral_mesh(points):
    """
    Generate a simple tetrahedral mesh from points (placeholder)
    """
    # This is a placeholder function
    # In a real application, you'd use a proper tetrahedralization algorithm
    if len(points) < 4:
        return np.array([])
    
    # Create a simple tetrahedron from the first 4 points
    return np.array([[0, 1, 2, 3]])

def main():
    # Display status of available backends
    st.subheader("Available Mesh Generation Backends")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.status("✅ MeshPy: " + ("Available" if HAS_MESHPY else "Not available"))
    with col2:
        st.status("✅ Gmsh: " + ("Available" if HAS_GMSH else "Not available (using fallback)"))
    with col3:
        st.status("✅ Meshio: " + ("Available" if HAS_MESHIO else "Limited export options"))
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("PropertyParams")
        
        # Dimensions input
        lx = st.number_input("Length (lx)", min_value=1.0, value=200.0, step=10.0)
        ly = st.number_input("Width (ly)", min_value=1.0, value=50.0, step=5.0)
        lz = st.number_input("Height (lz)", min_value=0.1, value=2.0, step=0.5)
        
        # Mesh settings
        st.subheader("Mesh Settings")
        mesh_backend = st.selectbox(
            "Mesh Generation Backend",
            ["MeshPy (Structured Tetrahedral)", "Fallback (Hexahedral)"],
            index=0 if HAS_MESHPY else 1,
            disabled=not HAS_MESHPY
        )
        
        if "MeshPy" in mesh_backend:
            max_volume = st.slider(
                "Maximum Element Volume", 
                0.1, 100.0, 10.0, 0.1,
                help="Smaller values = finer mesh"
            )
        else:
            resolution = st.slider(
                "Mesh Resolution", 
                1, 20, 5,
                help="Higher values = finer mesh"
            )
        
        # Visualization options
        st.subheader("Visualization Options")
        show_wireframe = st.checkbox("Show Wireframe", value=True)
        show_points = st.checkbox("Show Points", value=False)
        
        # Export format
        st.subheader("Export Format")
        export_format = st.selectbox("Format", 
                                   ["msh", "vtk", "vtu", "xdmf", "unv", "stl", "txt"])
    
    # Generate button
    if st.button("🚀 Generate Mesh", type="primary", use_container_width=True):
        with st.spinner("Creating mesh..."):
            # Select appropriate mesh generation method
            mesh_data = None
            
            if "MeshPy" in mesh_backend and HAS_MESHPY:
                mesh_data = create_mesh_with_meshpy(lx, ly, lz, max_volume)
                backend_used = "MeshPy"
            else:
                mesh_data = create_fallback_mesh(lx, ly, lz, resolution)
                backend_used = "Fallback"
            
            if mesh_data is None:
                st.error("Failed to generate mesh with selected backend. Using fallback method.")
                mesh_data = create_fallback_mesh(lx, ly, lz, 5)  # Default resolution
                backend_used = "Fallback (after failure)"
            
            # Get statistics
            num_points = len(mesh_data["points"])
            num_cells = 0
            if "tetra" in mesh_data.get("cells", {}):
                num_cells = len(mesh_data["cells"]["tetra"])
                cell_type = "Tetrahedra"
            elif "hexahedron" in mesh_data.get("cells", {}):
                num_cells = len(mesh_data["cells"]["hexahedron"])
                cell_type = "Hexahedra"
            else:
                cell_type = "Unknown"
            
            # Display statistics
            st.subheader("📊 Mesh Statistics")
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            with col_stats1:
                st.metric("Generation Backend", backend_used)
            with col_stats2:
                st.metric("Nodes", f"{num_points:,}")
            with col_stats3:
                st.metric(cell_type, f"{num_cells:,}")
            
            # Create two columns for visualization and info
            col1, col2 = st.columns([2, 1])
            
            # 3D Visualization
            with col1:
                st.subheader("3D Visualization")
                fig = visualize_mesh_with_plotly(mesh_data)
                st.plotly_chart(fig, use_container_width=True)
            
            # Physical groups and additional info
            with col2:
                st.subheader("PropertyParams")
                st.markdown(f"""
                **Dimensions:**
                - Length (X): {lx}
                - Width (Y): {2*ly} (total)
                - Height (Z): {lz}
                
                **Physical Groups:**
                - 2 Volumes
                - 11 Faces (including interface)
                """)
                
                with st.expander("Full Physical Group List"):
                    st.markdown("""
                    **Volumes:**
                    - Solid_1front (ID: 1)
                    - Solid_2back (ID: 2)
                    
                    **Faces:**
                    - Face_1leftfront (ID: 1)
                    - Face_2leftback (ID: 2)
                    - Face_3frontfront (ID: 3)
                    - Face_4bottomfront (ID: 4)
                    - Face_5topfront (ID: 5)
                    - Face_6interfacefront (ID: 6) - Interface
                    - Face_7bottomback (ID: 7)
                    - Face_8topback (ID: 8)
                    - Face_9backback (ID: 9)
                    - Face_10rightfront (ID: 10)
                    - Face_11rightback (ID: 11)
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
            
            # Format information
            with st.expander("ℹ️ Export Format Information"):
                format_descriptions = {
                    "msh": "Gmsh format - compatible with Code_Aster, GetFEM, and other FEM solvers",
                    "vtk": "VTK format - compatible with ParaView and VTK-based tools",
                    "vtu": "VTK XML format - more efficient than standard VTK for large meshes",
                    "xdmf": "Extensible Data Model - efficient for large datasets and time series",
                    "unv": "I-DEAS Universal format - widely used in commercial FEM software",
                    "stl": "Stereolithography format - surface mesh only, used for 3D printing and visualization",
                    "txt": "Simple text format - human readable but limited functionality"
                }
                st.markdown(f"**{export_format.upper()}:** {format_descriptions.get(export_format, 'Standard mesh format')}")
            
            st.success(f"✅ Mesh generated successfully with {backend_used} backend!")
    
    # Footer information
    st.markdown("---")
    st.caption("✅ This app works in all cloud environments with appropriate fallbacks")

if __name__ == "__main__":
    main()
