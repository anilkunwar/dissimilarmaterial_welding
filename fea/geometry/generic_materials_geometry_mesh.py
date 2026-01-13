# app.py

import streamlit as st
import gmsh
import numpy as np
import os
import tempfile
import meshio
import pyvista as pv
from stpyvista import stpyvista
import io
import base64

# Set page config
st.set_page_config(
    page_title="3D Slab Mesh Generator",
    page_icon="📐",
    layout="wide"
)

def create_slab_geometry(lx, ly, lz, mesh_size_factor=1.0):
    """
    Create two slabs using Gmsh with physical groups for export
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # Disable terminal output
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
    gmsh.model.addPhysicalGroup(3, [solid1], name="Solid_1front")
    gmsh.model.addPhysicalGroup(3, [solid2], name="Solid_2back")
    
    # Faces
    gmsh.model.addPhysicalGroup(2, [s_left1], name="Face_1leftfront")
    gmsh.model.addPhysicalGroup(2, [s_left2], name="Face_2leftback")
    gmsh.model.addPhysicalGroup(2, [s_front1], name="Face_3frontfront")
    gmsh.model.addPhysicalGroup(2, [s_bottom1], name="Face_4bottomfront")
    gmsh.model.addPhysicalGroup(2, [s_top1], name="Face_5topfront")
    gmsh.model.addPhysicalGroup(2, [s_front2], name="Face_6interfacefront")
    gmsh.model.addPhysicalGroup(2, [s_bottom2], name="Face_7bottomback")
    gmsh.model.addPhysicalGroup(2, [s_top2], name="Face_8topback")
    gmsh.model.addPhysicalGroup(2, [s_back2], name="Face_9backback")
    gmsh.model.addPhysicalGroup(2, [s_right1], name="Face_10rightfront")
    gmsh.model.addPhysicalGroup(2, [s_right2], name="Face_11rightback")

    # Mesh settings
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size_factor)
    gmsh.option.setNumber("Mesh.MeshSizeMin", lc / 2)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc * 2)
    gmsh.option.setNumber("Mesh.ColorCarousel", 2)  # Color by physical group

    # Generate mesh
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")
    
    return gmsh

def visualize_mesh(gmsh_model, show_geometry=True, show_mesh=True):
    """Create 3D visualization using PyVista"""
    # Get mesh data from Gmsh
    nodes = gmsh_model.model.mesh.getNodes()
    node_coords = nodes[1].reshape(-1, 3)
    
    # Get tetrahedral elements (type 4 in GMSH)
    element_types, element_tags, element_nodes = gmsh_model.model.mesh.getElements(3, -1)
    
    if 4 in element_types:  # Tetrahedra
        idx = list(element_types).index(4)
        tet_nodes = element_nodes[idx]
        tets = tet_nodes.reshape(-1, 4) - 1  # Convert to 0-based indexing
        
        # Create PyVista unstructured grid
        cells = np.hstack([4*np.ones((tets.shape[0], 1), dtype=int), tets]).ravel()
        grid = pv.UnstructuredGrid(cells, [pv.CellType.TETRA]*tets.shape[0], node_coords)
        
        # Get surface for wireframe
        surface = grid.extract_surface()
        
        # Create plotter
        plotter = pv.Plotter(window_size=[800, 600])
        
        if show_mesh:
            # Add mesh with wireframe
            plotter.add_mesh(grid, color='lightblue', opacity=0.3, 
                           show_edges=True, edge_color='black', line_width=0.5)
        
        if show_geometry:
            # Add surface wireframe
            plotter.add_mesh(surface, color='gray', style='wireframe', 
                           line_width=1.5, opacity=0.7)
        
        # Add coordinate axes
        plotter.add_axes(line_width=5)
        
        # Set background and view
        plotter.set_background('white')
        plotter.view_isometric()
        
        return plotter
    return None

def get_mesh_stats(gmsh_model):
    """Get mesh statistics"""
    stats = {}
    
    # Get node count
    nodes = gmsh_model.model.mesh.getNodes()
    stats['nodes'] = len(nodes[0])
    
    # Get element counts by type
    element_types, element_tags, _ = gmsh_model.model.mesh.getElements(-1, -1)
    
    # Count elements by type
    stats['elements'] = {}
    for elem_type, tags in zip(element_types, element_tags):
        if elem_type == 1:  # Line
            stats['elements']['lines'] = len(tags)
        elif elem_type == 2:  # Triangle
            stats['elements']['triangles'] = len(tags)
        elif elem_type == 4:  # Tetrahedron
            stats['elements']['tetrahedra'] = len(tags)
        elif elem_type == 15:  # Point
            stats['elements']['points'] = len(tags)
    
    stats['total_elements'] = sum(len(tags) for tags in element_tags)
    
    return stats

def convert_to_format(gmsh_model, format_name, temp_dir):
    """Convert mesh to different formats using meshio"""
    # First write to temporary msh file
    temp_msh = os.path.join(temp_dir, "temp.msh")
    gmsh_model.write(temp_msh)
    
    # Read with meshio
    mesh = meshio.read(temp_msh)
    
    # Define output filename and format
    format_map = {
        "Gmsh (.msh)": ("mesh.msh", "gmsh"),
        "VTK (.vtk)": ("mesh.vtk", "vtk"),
        "VTU (.vtu)": ("mesh.vtu", "vtu"),
        "STL (.stl)": ("mesh.stl", "stl"),
        "XDMF (.xdmf)": ("mesh.xdmf", "xdmf"),
        "MED (.med)": ("mesh.med", "med"),
        "Nastran (.bdf)": ("mesh.bdf", "nastran"),
        "Abaqus (.inp)": ("mesh.inp", "abaqus"),
        "ANSYS (.cdb)": ("mesh.cdb", "ansys"),
        "FLAC3D (.f3grid)": ("mesh.f3grid", "flac"),
        "Tecplot (.dat)": ("mesh.dat", "tecplot"),
        "Plot3D (.xyz)": ("mesh.xyz", "plot3d"),
        "Exodus (.e)": ("mesh.e", "exodus"),
        "Dolfin XML (.xml)": ("mesh.xml", "dolfin-xml"),
        "OFF (.off)": ("mesh.off", "off"),
        "OBJ (.obj)": ("mesh.obj", "obj"),
        "PLY (.ply)": ("mesh.ply", "ply"),
        "SU2 (.su2)": ("mesh.su2", "su2"),
        "SVG (.svg)": ("mesh.svg", "svg"),
    }
    
    # UNV format requires special handling
    if format_name == "UNV (.unv)":
        output_file = os.path.join(temp_dir, "mesh.unv")
        # Write using Gmsh directly for UNV
        gmsh_model.write(output_file)
        with open(output_file, 'rb') as f:
            return f.read(), "mesh.unv"
    
    if format_name in format_map:
        filename, fmt = format_map[format_name]
        output_file = os.path.join(temp_dir, filename)
        
        try:
            mesh.write(output_file, file_format=fmt)
            with open(output_file, 'rb') as f:
                return f.read(), filename
        except Exception as e:
            st.warning(f"Could not export as {format_name}: {str(e)}")
            # Fallback to MSH
            with open(temp_msh, 'rb') as f:
                return f.read(), "mesh.msh"
    
    # Default to MSH
    with open(temp_msh, 'rb') as f:
        return f.read(), "mesh.msh"

def main():
    st.title("📐 3D Slab Mesh Generator")
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
            "UNV (.unv)",
            "XDMF (.xdmf)",
            "MED (.med)",
            "Nastran (.bdf)",
            "Abaqus (.inp)",
            "ANSYS (.cdb)",
            "FLAC3D (.f3grid)",
            "Tecplot (.dat)",
            "Plot3D (.xyz)",
            "Exodus (.e)",
            "Dolfin XML (.xml)",
            "OFF (.off)",
            "OBJ (.obj)",
            "PLY (.ply)",
            "SU2 (.su2)",
            "SVG (.svg)"
        ]
        selected_format = st.selectbox("Select export format", export_formats)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("3D Visualization")
        if st.button("🚀 Generate & Visualize", type="primary", use_container_width=True):
            with st.spinner("Generating geometry and mesh..."):
                try:
                    # Create temporary directory
                    with tempfile.TemporaryDirectory() as tmpdir:
                        # Generate mesh
                        gmsh_model = create_slab_geometry(lx, ly, lz, mesh_size_factor)
                        
                        # Get mesh statistics
                        stats = get_mesh_stats(gmsh_model)
                        
                        # Visualize
                        plotter = visualize_mesh(gmsh_model, show_geometry, show_mesh)
                        
                        if plotter:
                            # Display the 3D visualization
                            stpyvista(plotter, key="mesh_viewer")
                            plotter.close()
                        
                        # Display statistics
                        with col2:
                            st.subheader("📊 Mesh Statistics")
                            st.metric("Total Nodes", f"{stats['nodes']:,}")
                            st.metric("Total Elements", f"{stats['total_elements']:,}")
                            
                            if 'tetrahedra' in stats['elements']:
                                st.metric("Tetrahedra", f"{stats['elements']['tetrahedra']:,}")
                            if 'triangles' in stats['elements']:
                                st.metric("Triangles", f"{stats['elements']['triangles']:,}")
                            if 'lines' in stats['elements']:
                                st.metric("Lines", f"{stats['elements']['lines']:,}")
                            
                            st.markdown("---")
                            st.subheader("📦 Export Mesh")
                            
                            # Export button
                            mesh_data, filename = convert_to_format(gmsh_model, selected_format, tmpdir)
                            
                            # Get file extension for mime type
                            file_ext = filename.split('.')[-1]
                            mime_types = {
                                'msh': 'application/octet-stream',
                                'vtk': 'application/vnd.vtk',
                                'vtu': 'application/vnd.vtu',
                                'stl': 'application/vnd.ms-pki.stl',
                                'unv': 'application/octet-stream',
                                'xdmf': 'application/xdmf+xml',
                                'med': 'application/octet-stream',
                                'bdf': 'application/octet-stream',
                                'inp': 'application/octet-stream',
                                'cdb': 'application/octet-stream',
                                'f3grid': 'application/octet-stream',
                                'dat': 'text/plain',
                                'xyz': 'text/plain',
                                'e': 'application/octet-stream',
                                'xml': 'application/xml',
                                'off': 'application/octet-stream',
                                'obj': 'application/octet-stream',
                                'ply': 'application/octet-stream',
                                'su2': 'application/octet-stream',
                                'svg': 'image/svg+xml'
                            }
                            
                            mime_type = mime_types.get(file_ext, 'application/octet-stream')
                            
                            st.download_button(
                                label=f"📥 Download as {selected_format}",
                                data=mesh_data,
                                file_name=filename,
                                mime=mime_type,
                                use_container_width=True
                            )
                        
                        # Clean up
                        gmsh_model.finalize()
                        
                        st.success("✅ Mesh generated successfully!")
                        
                except Exception as e:
                    st.error(f"❌ Error during mesh generation: {str(e)}")
                    try:
                        gmsh.finalize()
                    except:
                        pass
    
    # Information section
    st.markdown("---")
    with st.expander("ℹ️ About this tool"):
        st.markdown("""
        ### Features:
        - **3D Visualization**: Interactive 3D view of geometry and mesh
        - **Multiple Export Formats**: Export to 20+ different mesh formats
        - **Physical Groups**: Preserves named volumes and faces for FEM analysis
        - **Adjustable Mesh Density**: Control mesh refinement
        
        ### Export Formats Supported:
        1. **Gmsh (.msh)** - Native Gmsh format
        2. **VTK (.vtk)** - Visualization Toolkit format
        3. **UNV (.unv)** - I-DEAS Universal format
        4. **STL (.stl)** - Stereolithography format
        5. **MED (.med)** - SALOME MED format
        6. **Abaqus (.inp)** - Abaqus input format
        7. **ANSYS (.cdb)** - ANSYS format
        8. **XDMF (.xdmf)** - eXtensible Data Model and Format
        9. **And 12+ more formats...**
        
        ### Physical Groups:
        The mesh includes named physical groups for easy boundary condition assignment:
        - **Volumes**: Solid_1front, Solid_2back
        - **Faces**: Face_1leftfront through Face_11rightback
        
        **Note**: For UNV format, the export uses Gmsh's native writer to ensure compatibility.
        """)
    
    st.caption("Powered by [Gmsh](https://gmsh.info) • [PyVista](https://pyvista.org) • [meshio](https://github.com/nschloe/meshio)")

if __name__ == "__main__":
    main()
