# app.py
import streamlit as st
import numpy as np
import os
import tempfile
import io
import base64
from contextlib import contextmanager

# Import visualization libraries with fallbacks
try:
    import pyvista as pv
    from stpyvista import stpyvista
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    st.warning("PyVista not available. 3D visualization disabled.")

# Try to import gmsh, but handle failure gracefully
HAS_GMSH = False
try:
    import gmsh
    HAS_GMSH = True
except ImportError as e:
    st.warning(f"Gmsh not available: {str(e)}")
    HAS_GMSH = False
except OSError as e:
    st.warning(f"System libraries for Gmsh missing: {str(e)}")
    HAS_GMSH = False

# Import meshio with fallback
HAS_MESHIO = False
try:
    import meshio
    HAS_MESHIO = True
except ImportError:
    st.warning("Meshio not available. Export functionality limited.")

@contextmanager
def gmsh_env():
    """Context manager for Gmsh to ensure proper initialization and cleanup"""
    if HAS_GMSH:
        gmsh.initialize([], False)  # No terminal output
        try:
            yield gmsh
        finally:
            gmsh.finalize()
    else:
        yield None

def create_fallback_mesh(lx, ly, lz, mesh_density=1.0):
    """Create a simple fallback mesh when Gmsh is not available"""
    # Create a simple structured hex mesh for the two slabs
    
    # Calculate divisions based on mesh density
    div_x = max(2, int(10 * mesh_density * lx / max(lx, ly, lz)))
    div_y = max(2, int(10 * mesh_density * ly / max(lx, ly, lz)))
    div_z = max(2, int(10 * mesh_density * lz / max(lx, ly, lz)))
    
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
    
    # Create a simple meshio Mesh object
    hex_cells = [("hexahedron", np.array(cells))]
    
    # Create physical groups (matching Gmsh names)
    point_data = {}
    cell_data = {
        "Solid_1front": [np.zeros(len(cells), dtype=int)],
        "Solid_2back": [np.zeros(len(cells), dtype=int)],
    }
    
    # Mark cells belonging to each solid
    interface_y = ly
    for idx, cell in enumerate(cells):
        # Get center of cell
        center_y = 0
        for node in cell:
            center_y += points[node][1]
        center_y /= 8
        
        if center_y < interface_y:
            cell_data["Solid_1front"][0][idx] = 1
        else:
            cell_data["Solid_2back"][0][idx] = 1
    
    mesh = meshio.Mesh(
        points=np.array(points),
        cells=hex_cells,
        cell_data=cell_data,
    )
    
    return mesh

def create_slab_geometry(lx, ly, lz, mesh_size_factor=1.0):
    """
    Create two slabs using Gmsh with physical groups for export
    Returns a meshio Mesh object
    """
    if not HAS_GMSH:
        return create_fallback_mesh(lx, ly, lz, mesh_size_factor)
    
    with gmsh_env() as gmsh:
        if gmsh is None:
            return create_fallback_mesh(lx, ly, lz, mesh_size_factor)
        
        gmsh.model.add("TwoSlabs")
        
        # Calculate mesh size based on dimensions
        base_size = min(lx, ly, lz) / 10.0
        lc = base_size * mesh_size_factor
        
        # === Solid_1front (front slab) ===
        p1 = gmsh.model.occ.addPoint(0, 0, 0, lc)
        p2 = gmsh.model.occ.addPoint(lx, 0, 0, lc)
        p3 = gmsh.model.occ.addPoint(lx, ly, 0, lc)
        p4 = gmsh.model.occ.addPoint(0, ly, 0, lc)
        p5 = gmsh.model.occ.addPoint(0, 0, lz, lc)
        p6 = gmsh.model.occ.addPoint(lx, 0, lz, lc)
        p7 = gmsh.model.occ.addPoint(lx, ly, lz, lc)
        p8 = gmsh.model.occ.addPoint(0, ly, lz, lc)
        
        # Lines
        l1 = gmsh.model.occ.addLine(p1, p2)
        l2 = gmsh.model.occ.addLine(p2, p3)
        l3 = gmsh.model.occ.addLine(p3, p4)
        l4 = gmsh.model.occ.addLine(p4, p1)
        l5 = gmsh.model.occ.addLine(p5, p6)
        l6 = gmsh.model.occ.addLine(p6, p7)
        l7 = gmsh.model.occ.addLine(p7, p8)
        l8 = gmsh.model.occ.addLine(p8, p5)
        l9 = gmsh.model.occ.addLine(p1, p5)
        l10 = gmsh.model.occ.addLine(p2, p6)
        l11 = gmsh.model.occ.addLine(p3, p7)
        l12 = gmsh.model.occ.addLine(p4, p8)
        
        # Surfaces
        bottom1 = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
        s_bottom1 = gmsh.model.occ.addPlaneSurface([bottom1])
        top1 = gmsh.model.occ.addCurveLoop([l5, l6, l7, l8])
        s_top1 = gmsh.model.occ.addPlaneSurface([top1])
        front1 = gmsh.model.occ.addCurveLoop([l1, l10, -l5, -l9])
        s_front1 = gmsh.model.occ.addPlaneSurface([front1])
        back1 = gmsh.model.occ.addCurveLoop([l3, l12, -l7, -l11])
        s_back1 = gmsh.model.occ.addPlaneSurface([back1])
        left1 = gmsh.model.occ.addCurveLoop([l4, l12, -l8, -l9])
        s_left1 = gmsh.model.occ.addPlaneSurface([left1])
        right1 = gmsh.model.occ.addCurveLoop([l2, l11, -l6, -l10])
        s_right1 = gmsh.model.occ.addPlaneSurface([right1])
        
        solid1_surf = gmsh.model.occ.addSurfaceLoop([s_bottom1, s_top1, s_front1, s_back1, s_left1, s_right1])
        solid1 = gmsh.model.occ.addVolume([solid1_surf])
        
        # === Solid_2back (back slab) ===
        p9 = gmsh.model.occ.addPoint(0, ly, 0, lc)
        p10 = gmsh.model.occ.addPoint(lx, ly, 0, lc)
        p11 = gmsh.model.occ.addPoint(lx, 2*ly, 0, lc)
        p12 = gmsh.model.occ.addPoint(0, 2*ly, 0, lc)
        p13 = gmsh.model.occ.addPoint(0, ly, lz, lc)
        p14 = gmsh.model.occ.addPoint(lx, ly, lz, lc)
        p15 = gmsh.model.occ.addPoint(lx, 2*ly, lz, lc)
        p16 = gmsh.model.occ.addPoint(0, 2*ly, lz, lc)
        
        l13 = gmsh.model.occ.addLine(p9, p10)
        l14 = gmsh.model.occ.addLine(p10, p11)
        l15 = gmsh.model.occ.addLine(p11, p12)
        l16 = gmsh.model.occ.addLine(p12, p9)
        l17 = gmsh.model.occ.addLine(p13, p14)
        l18 = gmsh.model.occ.addLine(p14, p15)
        l19 = gmsh.model.occ.addLine(p15, p16)
        l20 = gmsh.model.occ.addLine(p16, p13)
        l21 = gmsh.model.occ.addLine(p9, p13)
        l22 = gmsh.model.occ.addLine(p10, p14)
        l23 = gmsh.model.occ.addLine(p11, p15)
        l24 = gmsh.model.occ.addLine(p12, p16)
        
        bottom2 = gmsh.model.occ.addCurveLoop([l13, l14, l15, l16])
        s_bottom2 = gmsh.model.occ.addPlaneSurface([bottom2])
        top2 = gmsh.model.occ.addCurveLoop([l17, l18, l19, l20])
        s_top2 = gmsh.model.occ.addPlaneSurface([top2])
        front2 = gmsh.model.occ.addCurveLoop([l13, l22, -l17, -l21])
        s_front2 = gmsh.model.occ.addPlaneSurface([front2])
        back2 = gmsh.model.occ.addCurveLoop([l15, l24, -l19, -l23])
        s_back2 = gmsh.model.occ.addPlaneSurface([back2])
        left2 = gmsh.model.occ.addCurveLoop([l16, l24, -l20, -l21])
        s_left2 = gmsh.model.occ.addPlaneSurface([left2])
        right2 = gmsh.model.occ.addCurveLoop([l14, l23, -l18, -l22])
        s_right2 = gmsh.model.occ.addPlaneSurface([right2])
        
        solid2_surf = gmsh.model.occ.addSurfaceLoop([s_bottom2, s_top2, s_front2, s_back2, s_left2, s_right2])
        solid2 = gmsh.model.occ.addVolume([solid2_surf])
        
        gmsh.model.occ.synchronize()
        
        # === Physical Groups ===
        # Volumes
        gmsh.model.addPhysicalGroup(3, [solid1], 1, name="Solid_1front")
        gmsh.model.addPhysicalGroup(3, [solid2], 2, name="Solid_2back")
        
        # Faces
        gmsh.model.addPhysicalGroup(2, [s_left1], 1, name="Face_1leftfront")
        gmsh.model.addPhysicalGroup(2, [s_left2], 2, name="Face_2leftback")
        gmsh.model.addPhysicalGroup(2, [s_front1], 3, name="Face_3frontfront")
        gmsh.model.addPhysicalGroup(2, [s_bottom1], 4, name="Face_4bottomfront")
        gmsh.model.addPhysicalGroup(2, [s_top1], 5, name="Face_5topfront")
        gmsh.model.addPhysicalGroup(2, [s_front2], 6, name="Face_6interfacefront")
        gmsh.model.addPhysicalGroup(2, [s_bottom2], 7, name="Face_7bottomback")
        gmsh.model.addPhysicalGroup(2, [s_top2], 8, name="Face_8topback")
        gmsh.model.addPhysicalGroup(2, [s_back2], 9, name="Face_9backback")
        gmsh.model.addPhysicalGroup(2, [s_right1], 10, name="Face_10rightfront")
        gmsh.model.addPhysicalGroup(2, [s_right2], 11, name="Face_11rightback")
        
        # Mesh settings
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
        gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size_factor)
        gmsh.option.setNumber("Mesh.MeshSizeMin", lc / 2)
        gmsh.option.setNumber("Mesh.MeshSizeMax", lc * 2)
        
        # Generate mesh
        gmsh.model.mesh.generate(3)
        if gmsh.model.mesh.getDim() == 3:
            gmsh.model.mesh.optimize("Netgen")
        
        # Save to temporary file and read with meshio
        with tempfile.TemporaryDirectory() as tmpdir:
            msh_path = os.path.join(tmpdir, "mesh.msh")
            gmsh.write(msh_path)
            return meshio.read(msh_path)

def visualize_mesh(mesh, show_geometry=True, show_mesh=True):
    """Create 3D visualization using PyVista"""
    if not HAS_PYVISTA:
        st.warning("3D visualization requires PyVista")
        return None
    
    # Check if we have tetrahedra or hexahedra
    has_tetra = False
    has_hexa = False
    cells = None
    
    for cell_block in mesh.cells:
        if cell_block.type == "tetra":
            has_tetra = True
            cells = cell_block.data
        elif cell_block.type == "hexahedron":
            has_hexa = True
            cells = cell_block.data
    
    if not (has_tetra or has_hexa) or cells is None:
        st.warning("No 3D cells found in mesh for visualization")
        return None
    
    # Create PyVista grid
    if has_tetra:
        cell_type = pv.CellType.TETRA
        cell_npoints = 4
    else:  # hexahedra
        cell_type = pv.CellType.HEXAHEDRON
        cell_npoints = 8
    
    # Format cells array for PyVista
    cell_connectivity = np.hstack([
        np.full((cells.shape[0], 1), cell_npoints), 
        cells
    ]).ravel()
    
    grid = pv.UnstructuredGrid(
        cell_connectivity,
        np.array([cell_type] * cells.shape[0]),
        mesh.points
    )
    
    # Create plotter
    plotter = pv.Plotter(window_size=[800, 600])
    
    if show_mesh:
        # Add mesh with partial transparency
        plotter.add_mesh(grid, color='lightblue', opacity=0.3, 
                        show_edges=True, edge_color='gray', line_width=0.5)
    
    if show_geometry:
        # Add surface outline
        surface = grid.extract_surface()
        plotter.add_mesh(surface, color='black', style='wireframe', line_width=1.5)
    
    # Add coordinate axes
    plotter.add_axes(line_width=3)
    
    # Set background and view
    plotter.set_background('white')
    plotter.view_isometric()
    plotter.reset_camera()
    
    return plotter

def get_mesh_stats(mesh):
    """Get mesh statistics"""
    stats = {
        'nodes': len(mesh.points),
        'elements': {},
        'total_elements': 0
    }
    
    for cell_block in mesh.cells:
        cell_type = cell_block.type
        count = len(cell_block.data)
        stats['total_elements'] += count
        
        if cell_type == "tetra":
            stats['elements']['tetrahedra'] = count
        elif cell_type == "hexahedron":
            stats['elements']['hexahedra'] = count
        elif cell_type == "triangle":
            stats['elements']['triangles'] = count
        elif cell_type == "quad":
            stats['elements']['quadrangles'] = count
        elif cell_type == "line":
            stats['elements']['lines'] = count
    
    return stats

def convert_to_format(mesh, format_name, temp_dir):
    """Convert mesh to different formats"""
    format_map = {
        "Gmsh (.msh)": ("mesh.msh", "gmsh22"),
        "VTK (.vtk)": ("mesh.vtk", "vtk"),
        "VTU (.vtu)": ("mesh.vtu", "vtu"),
        "STL (.stl)": ("mesh.stl", "stl"),
        "XDMF (.xdmf)": ("mesh.xdmf", "xdmf"),
        "MED (.med)": ("mesh.med", "med"),
        "Nastran (.bdf)": ("mesh.bdf", "nastran"),
        "Abaqus (.inp)": ("mesh.inp", "abaqus"),
    }
    
    if format_name in format_map:
        filename, fmt = format_map[format_name]
        output_file = os.path.join(temp_dir, filename)
        
        try:
            mesh.write(output_file, file_format=fmt)
            with open(output_file, 'rb') as f:
                return f.read(), filename
        except Exception as e:
            st.warning(f"Could not export as {format_name}: {str(e)}")
            # Fallback to VTK
            fallback_file = os.path.join(temp_dir, "mesh.vtk")
            mesh.write(fallback_file, file_format="vtk")
            with open(fallback_file, 'rb') as f:
                return f.read(), "mesh.vtk"
    
    # Default to VTK
    output_file = os.path.join(temp_dir, "mesh.vtk")
    mesh.write(output_file, file_format="vtk")
    with open(output_file, 'rb') as f:
        return f.read(), "mesh.vtk"

def main():
    st.set_page_config(
        page_title="3D Slab Mesh Generator",
        page_icon="📐",
        layout="wide"
    )
    
    st.title("📐 3D Slab Mesh Generator")
    
    # Status indicators
    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        st.status("✅ Gmsh: " + ("Available" if HAS_GMSH else "Not available (using fallback)"))
    with status_col2:
        st.status("✅ Meshio: " + ("Available" if HAS_MESHIO else "Not available"))
    with status_col3:
        st.status("✅ PyVista: " + ("Available" if HAS_PYVISTA else "Not available"))
    
    st.markdown("""
    Generate a 3D mesh of two adjacent slabs with comprehensive visualization and export capabilities.
    - **Slab 1 (Solid_1front)**: `(0,0,0)` → `(lx, ly, lz)`
    - **Slab 2 (Solid_2back)**:  `(0,ly,0)` → `(lx, 2·ly, lz)`
    """)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            lx = st.number_input("Length X", min_value=0.1, value=200.0, step=10.0, help="Length in X direction")
        with col2:
            ly = st.number_input("Width Y", min_value=0.1, value=50.0, step=5.0, help="Width in Y direction")
        
        lz = st.number_input("Height Z", min_value=0.1, value=2.0, step=0.5, help="Height in Z direction")
        
        st.markdown("---")
        
        mesh_size_factor = st.slider("Mesh Density", 0.1, 5.0, 1.0, 0.1, 
                                   help="Smaller values = finer mesh")
        
        st.markdown("---")
        
        # Visualization options
        st.subheader("🎨 Visualization")
        show_geometry = st.checkbox("Show Geometry", value=True)
        show_mesh = st.checkbox("Show Mesh", value=True)
        
        st.markdown("---")
        
        # Export format selection
        st.subheader("📤 Export Format")
        export_formats = [
            "Gmsh (.msh)",
            "VTK (.vtk)",
            "VTU (.vtu)",
            "STL (.stl)",
            "XDMF (.xdmf)",
            "MED (.med)",
            "Nastran (.bdf)",
            "Abaqus (.inp)",
        ]
        selected_format = st.selectbox("Select export format", export_formats)
    
    # Main content area - Generate button
    if st.button("🚀 Generate Mesh", type="primary", use_container_width=True):
        with st.spinner("Generating geometry and mesh..."):
            try:
                # Create temporary directory
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Generate mesh
                    mesh = create_slab_geometry(lx, ly, lz, mesh_size_factor)
                    
                    # Get mesh statistics
                    stats = get_mesh_stats(mesh)
                    
                    # Create two columns for visualization and stats
                    col1, col2 = st.columns([2, 1])
                    
                    # Visualization
                    with col1:
                        st.subheader("3D Visualization")
                        if HAS_PYVISTA:
                            plotter = visualize_mesh(mesh, show_geometry, show_mesh)
                            if plotter:
                                stpyvista(plotter, key="mesh_viewer")
                                plotter.close()
                        else:
                            st.info("Install PyVista for 3D visualization")
                    
                    # Statistics and export
                    with col2:
                        st.subheader("📊 Mesh Statistics")
                        st.metric("Total Nodes", f"{stats['nodes']:,}")
                        st.metric("Total Elements", f"{stats['total_elements']:,}")
                        
                        if 'tetrahedra' in stats['elements']:
                            st.metric("Tetrahedra", f"{stats['elements']['tetrahedra']:,}")
                        if 'hexahedra' in stats['elements']:
                            st.metric("Hexahedra", f"{stats['elements']['hexahedra']:,}")
                        if 'triangles' in stats['elements']:
                            st.metric("Triangles", f"{stats['elements']['triangles']:,}")
                        
                        st.markdown("---")
                        st.subheader("📦 Export Mesh")
                        
                        # Export button
                        mesh_data, filename = convert_to_format(mesh, selected_format, tmpdir)
                        
                        # Get file extension for mime type
                        file_ext = filename.split('.')[-1]
                        mime_types = {
                            'msh': 'application/octet-stream',
                            'vtk': 'application/vnd.vtk',
                            'vtu': 'application/vnd.vtu',
                            'stl': 'application/sla',
                            'xdmf': 'application/xdmf+xml',
                            'med': 'application/octet-stream',
                            'bdf': 'text/plain',
                            'inp': 'text/plain',
                        }
                        
                        mime_type = mime_types.get(file_ext, 'application/octet-stream')
                        
                        st.download_button(
                            label=f"📥 Download as {selected_format}",
                            data=mesh_data,
                            file_name=filename,
                            mime=mime_type,
                            use_container_width=True
                        )
                    
                    st.success("✅ Mesh generated successfully!")
                    if not HAS_GMSH:
                        st.info("Note: Using simplified fallback mesh algorithm since Gmsh is not available in this environment.")
            
            except Exception as e:
                st.error(f"❌ Error during mesh generation: {str(e)}")
                st.exception(e)
    
    # Information section
    st.markdown("---")
    with st.expander("ℹ️ About this tool"):
        st.markdown("""
        ### Cloud-Compatible Mesh Generator
        
        This application works in cloud environments like Streamlit Cloud where Gmsh might not be fully available:
        
        - ✅ **When Gmsh is available**: Uses full Gmsh capabilities for high-quality tetrahedral meshes
        - ✅ **When Gmsh is unavailable**: Uses a pure Python fallback to generate hexahedral meshes
        - ✅ **Visualization**: Uses PyVista for interactive 3D visualization when available
        
        ### Physical Groups:
        The mesh includes named physical groups for easy boundary condition assignment:
        - **Volumes**: Solid_1front, Solid_2back
        - **Faces**: Face_1leftfront through Face_11rightback
        
        ### Running Locally with Full Features:
        
        To use all features locally, install the required packages:
        ```bash
        pip install gmsh meshio pyvista stpyvista streamlit numpy
        ```
        
        Then run:
        ```bash
        streamlit run app.py
        ```
        """)

if __name__ == "__main__":
    main()
