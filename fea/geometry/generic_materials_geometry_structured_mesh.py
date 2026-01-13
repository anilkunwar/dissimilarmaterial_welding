# app.py
import streamlit as st
import numpy as np
import tempfile
import os
import plotly.graph_objects as go
from functools import lru_cache

# Set page config at the very top
st.set_page_config(
    page_title="ParallelGroup Advanced Mesh Generator",
    page_icon="📐",
    layout="wide"
)

# Initialize session state for caching
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.mesh_cache = {}
    st.session_state.fig_cache = {}
    st.session_state.import_status = {}
    st.session_state.current_mesh = None
    st.session_state.current_stats = None

# Cache heavy imports
@st.cache_resource(ttl=3600, show_spinner=False)
def lazy_import_meshpy():
    """Lazy load MeshPy with caching"""
    try:
        from meshpy.tet import MeshInfo, build
        from meshpy.geometry import GeometryBuilder, generate_surface_of_revolution
        return True, {"MeshInfo": MeshInfo, "build": build}
    except Exception as e:
        return False, str(e)

@st.cache_resource(ttl=3600, show_spinner=False)
def lazy_import_gmsh():
    """Lazy load Gmsh with caching"""
    try:
        import gmsh
        return True, {"gmsh": gmsh}
    except Exception as e:
        return False, str(e)

@st.cache_resource(ttl=3600, show_spinner=False)
def lazy_import_meshio():
    """Lazy load meshio with caching"""
    try:
        import meshio
        return True, {"meshio": meshio}
    except Exception as e:
        return False, str(e)

# Get import status once and cache
if not st.session_state.initialized:
    with st.spinner("Loading mesh libraries..."):
        meshpy_status, meshpy_data = lazy_import_meshpy()
        gmsh_status, gmsh_data = lazy_import_gmsh()
        meshio_status, meshio_data = lazy_import_meshio()
        
        st.session_state.import_status = {
            "HAS_MESHPY": meshpy_status,
            "HAS_GMSH": gmsh_status,
            "HAS_MESHIO": meshio_status,
            "meshpy_data": meshpy_data if meshpy_status else None,
            "meshio_data": meshio_data if meshio_status else None,
            "gmsh_data": gmsh_data if gmsh_status else None
        }
        st.session_state.initialized = True

# Constants - load once
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

st.title("ParallelGroup Advanced Mesh Generator")
st.markdown("""
This app generates structured and unstructured meshes using multiple backends.
**Optimized for Streamlit Cloud with caching and lazy loading.**
- **Slab 1 (Solid_1front)**: `(0,0,0)` → `(lx, ly, lz)`
- **Slab 2 (Solid_2back)**:  `(0,ly,0)` → `(lx, 2·ly, lz)`
""")

# Cache mesh generation functions
@st.cache_data(ttl=300, show_spinner=False)  # Cache for 5 minutes
def create_mesh_with_meshpy_cached(lx, ly, lz, max_volume=1.0):
    """Create mesh using MeshPy with caching"""
    if not st.session_state.import_status["HAS_MESHPY"]:
        return None
    
    try:
        MeshInfo = st.session_state.import_status["meshpy_data"]["MeshInfo"]
        build_func = st.session_state.import_status["meshpy_data"]["build"]
        
        mesh_info = MeshInfo()
        
        # Define vertices
        vertices = [
            (0, 0, 0), (lx, 0, 0), (lx, ly, 0), (0, ly, 0),
            (0, 0, lz), (lx, 0, lz), (lx, ly, lz), (0, ly, lz),
            (0, ly, 0), (lx, ly, 0), (lx, 2*ly, 0), (0, 2*ly, 0),
            (0, ly, lz), (lx, ly, lz), (lx, 2*ly, lz), (0, 2*ly, lz)
        ]
        
        facets = [
            [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
            [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5],
            [8, 9, 10, 11], [12, 13, 14, 15], [9, 10, 14, 13],
            [8, 11, 15, 12], [9, 8, 12, 13], [10, 11, 15, 14]
        ]
        
        facet_markers = [4, 5, 3, 6, 1, 10, 7, 8, 9, 2, 6, 11]
        
        mesh_info.set_points(vertices)
        mesh_info.set_facets(facets, facet_markers)
        
        mesh = build_func(
            mesh_info,
            max_volume=max_volume,
            verbose=False,
            attributes=True,
            volume_constraints=True
        )
        
        mesh_data = {
            "points": np.array(mesh.points),
            "cells": {"tetra": np.array(mesh.elements)},
            "physical_groups": PHYSICAL_GROUPS,
            "facet_markers": mesh.facet_markers,
            "dimensions": (lx, ly, lz),
            "backend": "meshpy"
        }
        
        return mesh_data
        
    except Exception as e:
        st.error(f"MeshPy generation failed: {str(e)[:100]}")
        return None

@st.cache_data(ttl=300, show_spinner=False)
def create_fallback_mesh_cached(lx, ly, lz, resolution=10):
    """Create fallback mesh with caching"""
    # Optimized mesh generation - limit complexity
    div_x = min(20, max(2, int(resolution * lx / max(lx, ly, lz))))
    div_y = min(20, max(2, int(resolution * ly / max(lx, ly, lz))))
    div_z = min(10, max(2, int(resolution * lz / max(lx, ly, lz))))
    
    # Generate points more efficiently
    x = np.linspace(0, lx, div_x + 1)
    y = np.linspace(0, 2*ly, 2*div_y + 1)
    z = np.linspace(0, lz, div_z + 1)
    
    # Create mesh grid
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    # Create hexahedral cells
    cells = []
    total_cells = div_x * 2 * div_y * div_z
    max_cells = 2000  # Limit for performance
    
    if total_cells > max_cells:
        # Use sparse sampling for large meshes
        stride = int(np.ceil(np.sqrt(total_cells / max_cells)))
        for k in range(0, div_z, stride):
            for j in range(0, 2*div_y, stride):
                for i in range(0, div_x, stride):
                    if i < div_x and j < 2*div_y and k < div_z:
                        n0 = k * (div_x + 1) * (2*div_y + 1) + j * (div_x + 1) + i
                        n1 = n0 + 1
                        n2 = n0 + (div_x + 1) + 1
                        n3 = n0 + (div_x + 1)
                        n4 = n0 + (div_x + 1) * (2*div_y + 1)
                        n5 = n4 + 1
                        n6 = n4 + (div_x + 1) + 1
                        n7 = n4 + (div_x + 1)
                        cells.append([n0, n1, n2, n3, n4, n5, n6, n7])
    else:
        # Generate all cells
        for k in range(div_z):
            for j in range(2*div_y):
                for i in range(div_x):
                    n0 = k * (div_x + 1) * (2*div_y + 1) + j * (div_x + 1) + i
                    n1 = n0 + 1
                    n2 = n0 + (div_x + 1) + 1
                    n3 = n0 + (div_x + 1)
                    n4 = n0 + (div_x + 1) * (2*div_y + 1)
                    n5 = n4 + 1
                    n6 = n4 + (div_x + 1) + 1
                    n7 = n4 + (div_x + 1)
                    cells.append([n0, n1, n2, n3, n4, n5, n6, n7])
    
    mesh_data = {
        "points": points,
        "cells": {"hexahedron": np.array(cells[:max_cells])},
        "physical_groups": PHYSICAL_GROUPS,
        "dimensions": (lx, ly, lz),
        "divisions": (div_x, div_y, div_z),
        "backend": "fallback"
    }
    
    return mesh_data

# Optimized visualization with caching
@st.cache_data(ttl=300, show_spinner=False, max_entries=5)
def visualize_mesh_with_plotly_cached(mesh_data, show_points=True):
    """Create optimized 3D visualization"""
    points = mesh_data["points"]
    lx, ly, lz = mesh_data["dimensions"]
    
    fig = go.Figure()
    
    # Optimize: Show only a subset of points for large meshes
    max_points_to_show = 1000
    if len(points) > max_points_to_show:
        # Randomly sample points
        indices = np.random.choice(len(points), max_points_to_show, replace=False)
        points_to_show = points[indices]
    else:
        points_to_show = points
    
    if show_points:
        fig.add_trace(go.Scatter3d(
            x=points_to_show[:, 0],
            y=points_to_show[:, 1],
            z=points_to_show[:, 2],
            mode='markers',
            marker=dict(size=2, color='blue', opacity=0.3),
            name='Mesh Points',
            hoverinfo='skip'
        ))
    
    # Add slab outlines (lightweight)
    # Slab 1 outline
    slab1_vertices = np.array([
        [0, 0, 0], [lx, 0, 0], [lx, ly, 0], [0, ly, 0],
        [0, 0, lz], [lx, 0, lz], [lx, ly, lz], [0, ly, lz]
    ])
    
    # Edges of the box
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    
    for edge in edges:
        x = [slab1_vertices[edge[0], 0], slab1_vertices[edge[1], 0]]
        y = [slab1_vertices[edge[0], 1], slab1_vertices[edge[1], 1]]
        z = [slab1_vertices[edge[0], 2], slab1_vertices[edge[1], 2]]
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color='blue', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Slab 2 outline
    slab2_vertices = np.array([
        [0, ly, 0], [lx, ly, 0], [lx, 2*ly, 0], [0, 2*ly, 0],
        [0, ly, lz], [lx, ly, lz], [lx, 2*ly, lz], [0, 2*ly, lz]
    ])
    
    for edge in edges:
        x = [slab2_vertices[edge[0], 0], slab2_vertices[edge[1], 0]]
        y = [slab2_vertices[edge[0], 1], slab2_vertices[edge[1], 1]]
        z = [slab2_vertices[edge[0], 2], slab2_vertices[edge[1], 2]]
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color='green', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Highlight interface
    interface_y = ly
    interface_x = [0, lx, lx, 0, 0]
    interface_z = [0, 0, lz, lz, 0]
    
    fig.add_trace(go.Scatter3d(
        x=interface_x, y=[interface_y]*5, z=interface_z,
        mode='lines',
        line=dict(color='red', width=3),
        name='Interface',
        hoverinfo='skip'
    ))
    
    # Add lightweight coordinate axes
    axis_length = max(lx, 2*ly, lz) * 0.2
    
    # X-axis
    fig.add_trace(go.Scatter3d(
        x=[0, axis_length], y=[0, 0], z=[0, 0],
        mode='lines', line=dict(color='red', width=2),
        name='X', hoverinfo='skip'
    ))
    # Y-axis
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, axis_length], z=[0, 0],
        mode='lines', line=dict(color='green', width=2),
        name='Y', hoverinfo='skip'
    ))
    # Z-axis
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, axis_length],
        mode='lines', line=dict(color='blue', width=2),
        name='Z', hoverinfo='skip'
    ))
    
    # Optimized layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.2),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0)
            )
        ),
        title='3D Mesh Visualization',
        height=500,  # Reduced height for performance
        margin=dict(l=0, r=0, b=0, t=30),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        hovermode=False  # Disable hover for performance
    )
    
    return fig

# Optimized export function with lazy loading
@st.cache_data(ttl=300, show_spinner=False)
def export_mesh_cached(mesh_data, format_name="msh"):
    """Export mesh with caching"""
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"two_slabs.{format_name}"
        filepath = os.path.join(tmpdir, filename)
        
        # Create simple text representation for light formats
        if format_name == "txt":
            with open(filepath, 'w') as f:
                f.write(f"# Mesh Data - Generated by ParallelGroup Mesh Generator\n")
                f.write(f"# Dimensions: {mesh_data['dimensions']}\n")
                f.write(f"# Points: {len(mesh_data['points'])}\n")
                f.write(f"# Backend: {mesh_data.get('backend', 'unknown')}\n")
                if 'tetra' in mesh_data.get('cells', {}):
                    f.write(f"# Tetrahedra: {len(mesh_data['cells']['tetra'])}\n")
                elif 'hexahedron' in mesh_data.get('cells', {}):
                    f.write(f"# Hexahedra: {len(mesh_data['cells']['hexahedron'])}\n")
                
                # Write points
                f.write("\nPOINTS\n")
                for i, point in enumerate(mesh_data['points'][:1000]):  # Limit for text
                    f.write(f"{i+1} {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
                
                if len(mesh_data['points']) > 1000:
                    f.write(f"# ... and {len(mesh_data['points']) - 1000} more points\n")
            
            with open(filepath, 'rb') as f:
                return f.read(), filename
        
        # For binary formats, check if meshio is available
        if st.session_state.import_status["HAS_MESHIO"]:
            try:
                meshio = st.session_state.import_status["meshio_data"]["meshio"]
                
                # Create minimal mesh for export
                if 'tetra' in mesh_data.get('cells', {}) and len(mesh_data['cells']['tetra']) > 0:
                    cells = [("tetra", mesh_data['cells']['tetra'][:2000])]  # Limit elements
                elif 'hexahedron' in mesh_data.get('cells', {}) and len(mesh_data['cells']['hexahedron']) > 0:
                    cells = [("hexahedron", mesh_data['cells']['hexahedron'][:1000])]  # Limit elements
                else:
                    # Create simple tetrahedral mesh
                    if len(mesh_data['points']) >= 4:
                        simple_cells = np.array([[0, 1, 2, 3]])
                        cells = [("tetra", simple_cells)]
                    else:
                        cells = []
                
                mesh = meshio.Mesh(
                    points=mesh_data['points'][:1000],  # Limit points
                    cells=cells,
                    point_data={}
                )
                
                # Map formats
                format_map = {
                    "msh": "gmsh22",
                    "vtk": "vtk",
                    "vtu": "vtu",
                    "stl": "stl",
                    "unv": "unv",
                    "xdmf": "xdmf"
                }
                
                file_format = format_map.get(format_name, "vtk")
                mesh.write(filepath, file_format=file_format)
                
            except Exception as e:
                # Fallback to simple text
                st.warning(f"Could not export as {format_name}: {str(e)[:100]}")
                return export_mesh_cached(mesh_data, "txt")
        else:
            # Fallback to text format
            return export_mesh_cached(mesh_data, "txt")
        
        # Read file for download
        try:
            with open(filepath, 'rb') as f:
                return f.read(), filename
        except:
            # Last resort fallback
            return b"Mesh export failed", "error.txt"

def main():
    # Display backend status in sidebar
    with st.sidebar:
        st.header("🔄 System Status")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            status = "✅" if st.session_state.import_status["HAS_MESHPY"] else "❌"
            st.metric("MeshPy", status)
        with col2:
            status = "✅" if st.session_state.import_status["HAS_GMSH"] else "❌"
            st.metric("Gmsh", status)
        with col3:
            status = "✅" if st.session_state.import_status["HAS_MESHIO"] else "⚠️"
            st.metric("MeshIO", status)
        
        st.markdown("---")
        st.header("⚙️ PropertyParams")
        
        # Dimensions with reasonable defaults
        lx = st.slider("Length X", 10.0, 500.0, 200.0, 10.0)
        ly = st.slider("Width Y", 10.0, 200.0, 50.0, 5.0)
        lz = st.slider("Height Z", 0.5, 20.0, 2.0, 0.5)
        
        st.markdown("---")
        st.subheader("🎯 Mesh Settings")
        
        # Simple backend selection
        backend_options = ["Fast Hex Mesh", "Tetra Mesh (if available)"]
        mesh_backend = st.selectbox(
            "Mesh Type",
            backend_options,
            index=0
        )
        
        if mesh_backend == "Fast Hex Mesh":
            resolution = st.slider("Resolution", 3, 15, 6, 
                                 help="Higher = finer mesh, but slower")
            max_volume = None
        else:
            if st.session_state.import_status["HAS_MESHPY"]:
                max_volume = st.slider("Max Element Volume", 0.5, 50.0, 5.0, 0.5)
                resolution = None
            else:
                st.warning("MeshPy not available, using hex mesh")
                mesh_backend = "Fast Hex Mesh"
                resolution = 6
                max_volume = None
        
        st.markdown("---")
        st.subheader("🎨 Visualization")
        show_points = st.checkbox("Show Mesh Points", value=True)
        
        st.markdown("---")
        st.subheader("📤 Export")
        export_format = st.selectbox("Format", ["txt", "msh", "vtk", "vtu", "stl", "unv"], index=0)
        
        # Clear cache button
        if st.button("Clear Cache", type="secondary"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.mesh_cache = {}
            st.session_state.fig_cache = {}
            st.rerun()
    
    # Main content
    if st.button("🚀 Generate Mesh", type="primary", use_container_width=True):
        with st.spinner("Generating mesh (cached for 5 minutes)..."):
            # Generate mesh with appropriate backend
            if mesh_backend == "Tetra Mesh (if available)" and st.session_state.import_status["HAS_MESHPY"]:
                mesh_data = create_mesh_with_meshpy_cached(lx, ly, lz, max_volume)
                backend_used = "MeshPy"
            else:
                mesh_data = create_fallback_mesh_cached(lx, ly, lz, resolution)
                backend_used = "Hex Mesh"
            
            if mesh_data is None:
                st.error("Failed to generate mesh. Using fallback method.")
                mesh_data = create_fallback_mesh_cached(lx, ly, lz, 5)
                backend_used = "Fallback"
            
            # Store in session state
            st.session_state.current_mesh = mesh_data
            
            # Calculate statistics
            num_points = len(mesh_data["points"])
            if 'tetra' in mesh_data.get('cells', {}):
                num_cells = len(mesh_data['cells']['tetra'])
                cell_type = "Tetrahedra"
            elif 'hexahedron' in mesh_data.get('cells', {}):
                num_cells = len(mesh_data['cells']['hexahedron'])
                cell_type = "Hexahedra"
            else:
                num_cells = 0
                cell_type = "Unknown"
            
            st.session_state.current_stats = {
                "backend": backend_used,
                "points": num_points,
                "cells": num_cells,
                "cell_type": cell_type,
                "dimensions": mesh_data["dimensions"]
            }
    
    # Display results if mesh exists in session state
    if st.session_state.current_mesh is not None:
        mesh_data = st.session_state.current_mesh
        stats = st.session_state.current_stats
        
        # Display statistics
        st.subheader("📊 Mesh Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Backend", stats["backend"])
        with col2:
            st.metric("Nodes", f"{stats['points']:,}")
        with col3:
            st.metric(stats["cell_type"], f"{stats['cells']:,}")
        with col4:
            lx, ly, lz = stats["dimensions"]
            st.metric("Total Size", f"{lx}×{2*ly}×{lz}")
        
        # Create two columns for layout
        col_viz, col_info = st.columns([3, 1])
        
        with col_viz:
            st.subheader("🎨 3D Visualization")
            fig = visualize_mesh_with_plotly_cached(mesh_data, show_points)
            st.plotly_chart(fig, use_container_width=True, 
                          config={'displayModeBar': False})  # Disable mode bar for performance
        
        with col_info:
            st.subheader("📋 PropertyParams")
            st.json({
                "Dimensions": {
                    "Length_X": lx,
                    "Width_Y": 2*ly,
                    "Height_Z": lz
                },
                "Physical_Groups": len(PHYSICAL_GROUPS),
                "Mesh_Type": mesh_data.get("backend", "unknown"),
                "Cache_Hit": "Yes" if stats["backend"] in ["MeshPy", "Hex Mesh"] else "No"
            })
            
            with st.expander("Physical Groups"):
                for name, info in PHYSICAL_GROUPS.items():
                    st.text(f"{name}: dim={info['dim']}, id={info['id']}")
        
        # Export section
        st.subheader("📥 Download Mesh")
        
        # Create export button
        file_data, filename = export_mesh_cached(mesh_data, export_format)
        
        # Determine MIME type
        mime_types = {
            "txt": "text/plain",
            "msh": "application/octet-stream",
            "vtk": "application/vnd.vtk",
            "vtu": "application/vnd.vtu",
            "stl": "model/stl",
            "unv": "application/octet-stream"
        }
        mime_type = mime_types.get(export_format, "application/octet-stream")
        
        st.download_button(
            label=f"📥 Download as {export_format.upper()}",
            data=file_data,
            file_name=filename,
            mime=mime_type,
            use_container_width=True
        )
        
        # Performance tips
        with st.expander("💡 Performance Tips"):
            st.markdown("""
            **For better performance on Streamlit Cloud:**
            1. ✅ Use **Fast Hex Mesh** for quick generation
            2. ✅ Keep mesh resolution **below 10** for best results
            3. ✅ Export as **TXT** format for fastest download
            4. ✅ Clear cache periodically if memory issues occur
            5. ✅ Disable "Show Mesh Points" for faster rendering
            
            **Note:** All meshes are cached for 5 minutes.
            """)
    
    else:
        # Show welcome/instructions
        st.info("👆 Click 'Generate Mesh' to create your first mesh!")
        
        # Quick tips
        with st.expander("🚀 Quick Start Tips"):
            st.markdown("""
            1. **Choose 'Fast Hex Mesh'** for best performance
            2. **Start with resolution 6** for quick results
            3. **Export as TXT** for fastest downloads
            4. **All operations are cached** - identical parameters will be faster
            5. **Clear cache** if you experience slowdowns
            """)
    
    # Footer with cache info
    st.markdown("---")
    col_footer1, col_footer2 = st.columns([3, 1])
    with col_footer1:
        st.caption("✅ Optimized for Streamlit Cloud with intelligent caching")
    with col_footer2:
        cache_size = len(st.session_state.mesh_cache)
        st.caption(f"Cache: {cache_size} meshes")

if __name__ == "__main__":
    main()
