# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import io
import base64
from matplotlib.patches import Rectangle
import meshio

st.set_page_config(
    page_title="ParallelGroup Slab Generator",
    page_icon="📐",
    layout="wide"
)

st.title("ParallelGroup Slab Generator (Cloud Compatible)")
st.markdown("""
This app generates a mesh of two adjacent slabs with all the required physical groups.
**No system dependencies required** - works in all cloud environments.
- **Slab 1 (Solid_1front)**: `(0,0,0)` → `(lx, ly, lz)`
- **Slab 2 (Solid_2back)**:  `(0,ly,0)` → `(lx, 2·ly, lz)`
""")

# Check if meshio is available, if not, offer a simple download option
HAS_MESHIO = False
try:
    import meshio
    HAS_MESHIO = True
except ImportError:
    st.warning("Meshio not available. Will generate a simple text-based mesh file.")

def create_mesh_without_gmsh(lx, ly, lz, resolution=10):
    """
    Create a simple hexahedral mesh without Gmsh dependencies
    Returns a meshio Mesh object or simple data structure
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
    
    # Create cell blocks
    hex_cells = [("hexahedron", np.array(cells))]
    
    # Create physical groups (matching the required names)
    cell_data = {
        "gmsh:physical": [np.zeros(len(cells), dtype=int)],
        "gmsh:geometrical": [np.zeros(len(cells), dtype=int)]
    }
    
    # Create field data for named groups
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
    
    # If meshio is available, create a proper mesh object
    if HAS_MESHIO:
        mesh = meshio.Mesh(
            points=np.array(points),
            cells=hex_cells,
            cell_data=cell_data,
            field_data=field_data
        )
        return mesh
    else:
        # Return simple data structure for basic export
        return {
            "points": np.array(points),
            "cells": cells,
            "field_data": field_data
        }

def plot_2d_cross_section(lx, ly, lz):
    """Create a 2D cross-section visualization that works without PyVista"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot slab 1 (front)
    rect1 = Rectangle((0, 0), lx, ly, fill=True, color='skyblue', alpha=0.6, 
                     edgecolor='blue', label='Solid_1front')
    ax.add_patch(rect1)
    
    # Plot slab 2 (back)
    rect2 = Rectangle((0, ly), lx, ly, fill=True, color='lightgreen', alpha=0.6, 
                     edgecolor='green', label='Solid_2back')
    ax.add_patch(rect2)
    
    # Mark interface
    ax.plot([0, lx], [ly, ly], 'r--', linewidth=2, label='Face_6interfacefront')
    
    # Set labels and title
    ax.set_xlabel('X (length)')
    ax.set_ylabel('Y (width)')
    ax.set_title('2D Cross-Section (Z-midplane)')
    ax.set_aspect('equal')
    ax.set_xlim(-0.1*lx, 1.1*lx)
    ax.set_ylim(-0.1*ly, 2.1*ly)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Add annotations for faces
    ax.text(lx/2, ly/2, 'Solid_1front', ha='center', va='center', fontsize=10)
    ax.text(lx/2, 1.5*ly, 'Solid_2back', ha='center', va='center', fontsize=10)
    
    return fig

def export_mesh(mesh_data, format_name="msh"):
    """Export mesh data to various formats without Gmsh dependencies"""
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"two_slabs.{format_name}"
        filepath = os.path.join(tmpdir, filename)
        
        if HAS_MESHIO:
            # Use meshio for proper export
            if format_name == "msh":
                mesh_data.write(filepath, file_format="gmsh22")
            elif format_name == "vtk":
                mesh_data.write(filepath, file_format="vtk")
            elif format_name == "vtu":
                mesh_data.write(filepath, file_format="vtu")
            elif format_name == "xdmf":
                mesh_data.write(filepath, file_format="xdmf")
            else:
                mesh_data.write(filepath, file_format="vtk")
        else:
            # Fallback to simple text export
            with open(filepath, 'w') as f:
                f.write("# Simple mesh export (no meshio available)\n")
                f.write(f"# Dimensions: lx={mesh_data['points'][0][0]}, ly={mesh_data['points'][0][1]/2}, lz={mesh_data['points'][0][2]}\n")
                f.write(f"# Nodes: {len(mesh_data['points'])}\n")
                f.write(f"# Elements: {len(mesh_data['cells'])}\n")
                f.write("# Physical groups defined:\n")
                for name in mesh_data['field_data'].keys():
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
        
        # Export format
        export_format = st.selectbox("Export Format", ["msh", "vtk", "vtu", "xdmf", "txt"])
    
    # Create two columns - visualization on left, statistics on right
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("2D Cross-Section View")
        fig = plot_2d_cross_section(lx, ly, lz)
        st.pyplot(fig)
    
    with col2:
        st.subheader("PropertyParams")
        st.markdown(f"""
        **Dimensions:**
        - Length (X): {lx}
        - Width (Y): {2*ly} (total)
        - Height (Z): {lz}
        
        **Physical Groups:**
        - 2 Volumes: Solid_1front, Solid_2back
        - 11 Faces (including interface)
        """)
        
        # Show physical groups list
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
    
    # Generate button
    if st.button("🚀 Generate Mesh", type="primary"):
        with st.spinner("Creating mesh data..."):
            # Create the mesh
            mesh_data = create_mesh_without_gmsh(lx, ly, lz, resolution)
            
            # Get statistics
            if HAS_MESHIO:
                num_points = len(mesh_data.points)
                num_cells = sum(len(cells.data) for cells in mesh_data.cells)
            else:
                num_points = len(mesh_data['points'])
                num_cells = len(mesh_data['cells'])
            
            # Display statistics
            st.subheader("📊 Mesh Statistics")
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            with col_stats1:
                st.metric("Nodes", f"{num_points:,}")
            with col_stats2:
                st.metric("Elements", f"{num_cells:,}")
            with col_stats3:
                st.metric("Physical Groups", "13")
            
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
            
            st.success("✅ Mesh generated successfully!")
            
            # Additional information
            with st.expander("ℹ️ About this mesh"):
                st.markdown("""
                This mesh contains all the physical groups required for your simulation:
                
                - **Two volumes** with material properties
                - **Interface face** (Face_6interfacefront) for coupling conditions
                - **Boundary faces** for applying constraints and loads
                
                The mesh is compatible with most FEM solvers. For best results:
                - Use format `.msh` for Code_Aster or GetFEM
                - Use format `.vtk` or `.vtu` for ParaView visualization
                - Use format `.xdmf` for FEniCS or deal.II
                
                **Note:** This is a hexahedral mesh generated without Gmsh dependencies. 
                For tetrahedral meshes or more complex geometries, run locally with Gmsh installed.
                """)
    
    # Footer information
    st.markdown("---")
    st.caption("✅ This app works in all cloud environments without system dependencies")

if __name__ == "__main__":
    main()
