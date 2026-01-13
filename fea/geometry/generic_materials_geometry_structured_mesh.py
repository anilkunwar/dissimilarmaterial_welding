# app.py
import streamlit as st
import numpy as np
import tempfile
import os
import plotly.graph_objects as go
import io
import time
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
    st.session_state.current_mesh_obj = None
    st.session_state.last_generation_time = None

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

@st.cache_resource(ttl=3600, show_spinner=False)
def lazy_import_pyvista():
    """Lazy load pyvista with caching"""
    try:
        import pyvista as pv
        return True, {"pv": pv}
    except Exception as e:
        return False, str(e)

# Get import status once and cache
if not st.session_state.initialized:
    with st.spinner("Loading mesh libraries..."):
        meshpy_status, meshpy_data = lazy_import_meshpy()
        gmsh_status, gmsh_data = lazy_import_gmsh()
        meshio_status, meshio_data = lazy_import_meshio()
        pyvista_status, pyvista_data = lazy_import_pyvista()
        
        st.session_state.import_status = {
            "HAS_MESHPY": meshpy_status,
            "HAS_GMSH": gmsh_status,
            "HAS_MESHIO": meshio_status,
            "HAS_PYVISTA": pyvista_status,
            "meshpy_data": meshpy_data if meshpy_status else None,
            "meshio_data": meshio_data if meshio_status else None,
            "gmsh_data": gmsh_data if gmsh_status else None,
            "pyvista_data": pyvista_data if pyvista_status else None
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

# Define all supported export formats with descriptions
EXPORT_FORMATS = {
    "Gmsh MSH (.msh)": {
        "extension": "msh",
        "mime": "application/octet-stream",
        "description": "Gmsh native format, compatible with most FEM solvers",
        "backend": "meshio"
    },
    "I-DEAS UNV (.unv)": {
        "extension": "unv",
        "mime": "application/octet-stream",
        "description": "I-DEAS Universal format for commercial FEM software",
        "backend": "meshio"
    },
    "VTK (.vtk)": {
        "extension": "vtk",
        "mime": "application/vnd.vtk",
        "description": "Visualization Toolkit format for ParaView",
        "backend": "meshio"
    },
    "VTU (.vtu)": {
        "extension": "vtu",
        "mime": "application/vnd.vtu",
        "description": "VTK XML format (compressed)",
        "backend": "meshio"
    },
    "STL (.stl)": {
        "extension": "stl",
        "mime": "model/stl",
        "description": "Stereolithography for 3D printing",
        "backend": "meshio"
    },
    "XDMF (.xdmf)": {
        "extension": "xdmf",
        "mime": "application/xdmf+xml",
        "description": "Extensible Data Model for large datasets",
        "backend": "meshio"
    },
    "MED (.med)": {
        "extension": "med",
        "mime": "application/octet-stream",
        "description": "SALOME MED format for Code_Aster",
        "backend": "meshio"
    },
    "Nastran BDF (.bdf)": {
        "extension": "bdf",
        "mime": "application/octet-stream",
        "description": "MSC Nastran bulk data format",
        "backend": "meshio"
    },
    "Abaqus INP (.inp)": {
        "extension": "inp",
        "mime": "application/octet-stream",
        "description": "Abaqus input file format",
        "backend": "meshio"
    },
    "ANSYS CDB (.cdb)": {
        "extension": "cdb",
        "mime": "application/octet-stream",
        "description": "ANSYS archive format",
        "backend": "meshio"
    },
    "PLY (.ply)": {
        "extension": "ply",
        "mime": "application/octet-stream",
        "description": "Polygon File Format",
        "backend": "meshio"
    },
    "OBJ (.obj)": {
        "extension": "obj",
        "mime": "application/octet-stream",
        "description": "Wavefront OBJ format",
        "backend": "meshio"
    },
    "Tecplot (.dat)": {
        "extension": "dat",
        "mime": "text/plain",
        "description": "Tecplot ASCII format",
        "backend": "meshio"
    },
    "Simple Text (.txt)": {
        "extension": "txt",
        "mime": "text/plain",
        "description": "Human-readable text format",
        "backend": "custom"
    },
    "OFF (.off)": {
        "extension": "off",
        "mime": "application/octet-stream",
        "description": "Object File Format",
        "backend": "meshio"
    },
    "SU2 (.su2)": {
        "extension": "su2",
        "mime": "application/octet-stream",
        "description": "SU2 CFD solver format",
        "backend": "meshio"
    },
    "Exodus (.exo)": {
        "extension": "exo",
        "mime": "application/octet-stream",
        "description": "Exodus II format",
        "backend": "meshio"
    }
}

st.title("ParallelGroup Advanced Mesh Generator")
st.markdown("""
This app generates structured and unstructured meshes using multiple backends.
**Optimized for Streamlit Cloud with caching and lazy loading.**
- **Slab 1 (Solid_1front)**: `(0,0,0)` → `(lx, ly, lz)`
- **Slab 2 (Solid_2back)**:  `(0,ly,0)` → `(lx, 2·ly, lz)`
""")

# Cache mesh generation functions
@st.cache_data(ttl=300, show_spinner=False)
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
        
        # Create meshio object for better export
        if st.session_state.import_status["HAS_MESHIO"]:
            meshio_module = st.session_state.import_status["meshio_data"]["meshio"]
            
            # Create cell blocks for meshio
            cells = [("tetra", mesh_data["cells"]["tetra"])]
            
            # Create meshio mesh object
            mesh_obj = meshio_module.Mesh(
                mesh_data["points"],
                cells,
                cell_sets={
                    "Solid_1front": [np.arange(0, len(mesh_data["cells"]["tetra"])//2)],
                    "Solid_2back": [np.arange(len(mesh_data["cells"]["tetra"])//2, len(mesh_data["cells"]["tetra"]))]
                }
            )
            mesh_data["meshio_object"] = mesh_obj
        
        return mesh_data
        
    except Exception as e:
        st.error(f"MeshPy generation failed: {str(e)[:100]}")
        return None

@st.cache_data(ttl=300, show_spinner=False)
def create_gmsh_mesh_cached(lx, ly, lz, mesh_size=1.0):
    """Create mesh using Gmsh with proper physical groups"""
    if not st.session_state.import_status["HAS_GMSH"]:
        return None
    
    try:
        gmsh_module = st.session_state.import_status["gmsh_data"]["gmsh"]
        
        # Initialize Gmsh
        gmsh_module.initialize()
        gmsh_module.model.add("TwoSlabs")
        
        # Set mesh size
        lc = min(lx, ly, lz) / 10.0 * mesh_size
        
        # Create first slab
        p1 = gmsh_module.model.occ.addPoint(0, 0, 0, lc)
        p2 = gmsh_module.model.occ.addPoint(lx, 0, 0, lc)
        p3 = gmsh_module.model.occ.addPoint(lx, ly, 0, lc)
        p4 = gmsh_module.model.occ.addPoint(0, ly, 0, lc)
        p5 = gmsh_module.model.occ.addPoint(0, 0, lz, lc)
        p6 = gmsh_module.model.occ.addPoint(lx, 0, lz, lc)
        p7 = gmsh_module.model.occ.addPoint(lx, ly, lz, lc)
        p8 = gmsh_module.model.occ.addPoint(0, ly, lz, lc)
        
        # Create second slab
        p9 = gmsh_module.model.occ.addPoint(0, ly, 0, lc)
        p10 = gmsh_module.model.occ.addPoint(lx, ly, 0, lc)
        p11 = gmsh_module.model.occ.addPoint(lx, 2*ly, 0, lc)
        p12 = gmsh_module.model.occ.addPoint(0, 2*ly, 0, lc)
        p13 = gmsh_module.model.occ.addPoint(0, ly, lz, lc)
        p14 = gmsh_module.model.occ.addPoint(lx, ly, lz, lc)
        p15 = gmsh_module.model.occ.addPoint(lx, 2*ly, lz, lc)
        p16 = gmsh_module.model.occ.addPoint(0, 2*ly, lz, lc)
        
        # Create volumes
        box1 = gmsh_module.model.occ.addBox(0, 0, 0, lx, ly, lz)
        box2 = gmsh_module.model.occ.addBox(0, ly, 0, lx, ly, lz)
        
        # Synchronize
        gmsh_module.model.occ.synchronize()
        
        # Add physical groups
        gmsh_module.model.addPhysicalGroup(3, [box1], 1)
        gmsh_module.model.setPhysicalName(3, 1, "Solid_1front")
        
        gmsh_module.model.addPhysicalGroup(3, [box2], 2)
        gmsh_module.model.setPhysicalName(3, 2, "Solid_2back")
        
        # Add surface physical groups
        surfaces = gmsh_module.model.getEntities(2)
        face_counter = 1
        for surf in surfaces:
            if surf[1] == 1:  # Left face of slab 1
                gmsh_module.model.addPhysicalGroup(2, [surf[0]], 10 + face_counter)
                gmsh_module.model.setPhysicalName(2, 10 + face_counter, f"Face_{face_counter}leftfront")
                face_counter += 1
            elif surf[1] == 6:  # Right face of slab 1
                gmsh_module.model.addPhysicalGroup(2, [surf[0]], 10 + face_counter)
                gmsh_module.model.setPhysicalName(2, 10 + face_counter, f"Face_{face_counter}rightfront")
                face_counter += 1
            # Add more face mappings as needed...
        
        # Generate mesh
        gmsh_module.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh_module.option.setNumber("Mesh.MeshSizeMin", lc/2)
        gmsh_module.option.setNumber("Mesh.MeshSizeMax", lc*2)
        
        gmsh_module.model.mesh.generate(3)
        
        # Get mesh data
        node_tags, node_coords, _ = gmsh_module.model.mesh.getNodes()
        node_coords = node_coords.reshape(-1, 3)
        
        # Get tetrahedral elements
        tetra_types, tetra_tags, tetra_nodes = gmsh_module.model.mesh.getElements(3, -1)
        
        if 4 in tetra_types:  # Tetrahedron element type
            idx = list(tetra_types).index(4)
            tet_nodes = tetra_nodes[idx]
            tets = tet_nodes.reshape(-1, 4) - 1  # Convert to 0-based indexing
            
            mesh_data = {
                "points": node_coords,
                "cells": {"tetra": tets},
                "physical_groups": PHYSICAL_GROUPS,
                "dimensions": (lx, ly, lz),
                "backend": "gmsh"
            }
            
            # Create meshio object
            if st.session_state.import_status["HAS_MESHIO"]:
                meshio_module = st.session_state.import_status["meshio_data"]["meshio"]
                cells = [("tetra", tets)]
                mesh_obj = meshio_module.Mesh(node_coords, cells)
                mesh_data["meshio_object"] = mesh_obj
                mesh_data["gmsh_model"] = gmsh_module  # Keep reference for UNV export
            
            return mesh_data
        
        gmsh_module.finalize()
        return None
        
    except Exception as e:
        st.error(f"Gmsh generation failed: {str(e)[:100]}")
        try:
            gmsh_module.finalize()
        except:
            pass
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
    
    cells_array = np.array(cells)
    
    mesh_data = {
        "points": points,
        "cells": {"hexahedron": cells_array},
        "physical_groups": PHYSICAL_GROUPS,
        "dimensions": (lx, ly, lz),
        "divisions": (div_x, div_y, div_z),
        "backend": "fallback"
    }
    
    # Create meshio object for better export
    if st.session_state.import_status["HAS_MESHIO"]:
        meshio_module = st.session_state.import_status["meshio_data"]["meshio"]
        
        # Split cells into two groups for the two slabs
        slab1_cells = []
        slab2_cells = []
        
        for idx, cell in enumerate(cells_array):
            # Get cell center y-coordinate
            cell_points = points[cell]
            center_y = np.mean(cell_points[:, 1])
            
            if center_y < ly:  # First slab
                slab1_cells.append(idx)
            else:  # Second slab
                slab2_cells.append(idx)
        
        # Create meshio mesh object with cell sets
        cells_list = [("hexahedron", cells_array)]
        cell_sets = {
            "Solid_1front": [np.array(slab1_cells)],
            "Solid_2back": [np.array(slab2_cells)]
        }
        
        mesh_obj = meshio_module.Mesh(points, cells_list, cell_sets=cell_sets)
        mesh_data["meshio_object"] = mesh_obj
    
    return mesh_data

def write_unv_format(mesh_data, filepath):
    """Write mesh in I-DEAS UNV format"""
    # UNV format specification:
    # - Section 2411: Nodes
    # - Section 2412: Elements
    # - Section 2467: Groups
    
    points = mesh_data["points"]
    
    with open(filepath, 'w') as f:
        # Write nodes (section 2411)
        f.write("    -1\n")
        f.write("  2411\n")
        for i, (x, y, z) in enumerate(points, 1):
            f.write(f"{i:10d}{1:10d}{1:10d}{1:10d}{1:10d}\n")
            f.write(f"    {x:13.6E}    {y:13.6E}    {z:13.6E}\n")
        f.write("    -1\n")
        
        # Write elements (section 2412)
        f.write("    -1\n")
        f.write("  2412\n")
        
        element_counter = 1
        
        # Write tetrahedral elements if available
        if "tetra" in mesh_data["cells"] and len(mesh_data["cells"]["tetra"]) > 0:
            tets = mesh_data["cells"]["tetra"]
            for tet in tets[:1000]:  # Limit for performance
                f.write(f"{element_counter:10d}{11:10d}{2:10d}{1:10d}{1:10d}{0:10d}{0:10d}\n")
                f.write(f"{tet[0]+1:10d}{tet[1]+1:10d}{tet[2]+1:10d}{tet[3]+1:10d}{0:10d}{0:10d}{0:10d}{0:10d}\n")
                element_counter += 1
        
        # Write hexahedral elements if available
        elif "hexahedron" in mesh_data["cells"] and len(mesh_data["cells"]["hexahedron"]) > 0:
            hexes = mesh_data["cells"]["hexahedron"]
            for hexa in hexes[:500]:  # Limit for performance
                f.write(f"{element_counter:10d}{6:10d}{2:10d}{1:10d}{1:10d}{0:10d}{0:10d}\n")
                f.write(f"{hexa[0]+1:10d}{hexa[1]+1:10d}{hexa[2]+1:10d}{hexa[3]+1:10d}")
                f.write(f"{hexa[4]+1:10d}{hexa[5]+1:10d}{hexa[6]+1:10d}{hexa[7]+1:10d}\n")
                element_counter += 1
        
        f.write("    -1\n")
        
        # Write groups (section 2467)
        f.write("    -1\n")
        f.write("  2467\n")
        
        # Write group definitions
        group_id = 1
        for name, info in PHYSICAL_GROUPS.items():
            f.write(f"{group_id:10d}{0:10d}{1:10d}{0:10d}{0:10d}\n")
            f.write(f"{name:80s}\n")
            group_id += 1
        
        f.write("    -1\n")
        
        # End of file
        f.write("    -1\n")

def write_msh_format(mesh_data, filepath):
    """Write mesh in Gmsh MSH format version 2.2"""
    points = mesh_data["points"]
    
    with open(filepath, 'w') as f:
        # Write header
        f.write("$MeshFormat\n")
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")
        
        # Write nodes
        f.write("$Nodes\n")
        f.write(f"{len(points)}\n")
        for i, (x, y, z) in enumerate(points, 1):
            f.write(f"{i} {x:.6f} {y:.6f} {z:.6f}\n")
        f.write("$EndNodes\n")
        
        # Write elements
        element_counter = 1
        elements_list = []
        
        # Collect tetrahedral elements
        if "tetra" in mesh_data["cells"] and len(mesh_data["cells"]["tetra"]) > 0:
            tets = mesh_data["cells"]["tetra"]
            for tet in tets[:1000]:  # Limit for performance
                elements_list.append((element_counter, 4, 2, 1, 1, 
                                     tet[0]+1, tet[1]+1, tet[2]+1, tet[3]+1))
                element_counter += 1
        
        # Collect hexahedral elements
        elif "hexahedron" in mesh_data["cells"] and len(mesh_data["cells"]["hexahedron"]) > 0:
            hexes = mesh_data["cells"]["hexahedron"]
            for hexa in hexes[:500]:  # Limit for performance
                elements_list.append((element_counter, 5, 2, 1, 1,
                                     hexa[0]+1, hexa[1]+1, hexa[2]+1, hexa[3]+1,
                                     hexa[4]+1, hexa[5]+1, hexa[6]+1, hexa[7]+1))
                element_counter += 1
        
        f.write("$Elements\n")
        f.write(f"{len(elements_list)}\n")
        
        for elem in elements_list:
            elem_str = " ".join(str(x) for x in elem)
            f.write(f"{elem_str}\n")
        
        f.write("$EndElements\n")
        
        # Write physical groups
        f.write("$PhysicalNames\n")
        f.write(f"{len(PHYSICAL_GROUPS)}\n")
        
        for name, info in PHYSICAL_GROUPS.items():
            f.write(f"{info['dim']} {info['id']} \"{name}\"\n")
        
        f.write("$EndPhysicalNames\n")

# Optimized visualization with caching
@st.cache_data(ttl=300, show_spinner=False, max_entries=5)
def visualize_mesh_with_plotly_cached(mesh_data, show_points=True, show_wireframe=True):
    """Create optimized 3D visualization"""
    points = mesh_data["points"]
    lx, ly, lz = mesh_data["dimensions"]
    
    fig = go.Figure()
    
    # Add mesh points if requested
    if show_points:
        # Sample points for performance
        max_points = min(500, len(points))
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            sample_points = points[indices]
        else:
            sample_points = points
        
        fig.add_trace(go.Scatter3d(
            x=sample_points[:, 0],
            y=sample_points[:, 1],
            z=sample_points[:, 2],
            mode='markers',
            marker=dict(size=2, color='blue', opacity=0.3),
            name='Mesh Points',
            hoverinfo='skip'
        ))
    
    # Add slab wireframes
    if show_wireframe:
        # Slab 1 wireframe
        slab1_vertices = np.array([
            [0, 0, 0], [lx, 0, 0], [lx, ly, 0], [0, ly, 0],
            [0, 0, lz], [lx, 0, lz], [lx, ly, lz], [0, ly, lz]
        ])
        
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
        
        # Slab 2 wireframe
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
    interface_x = [0, lx, lx, 0, 0]
    interface_z = [0, 0, lz, lz, 0]
    
    fig.add_trace(go.Scatter3d(
        x=interface_x,
        y=[ly]*5,
        z=interface_z,
        mode='lines',
        line=dict(color='red', width=3),
        name='Interface',
        hoverinfo='skip'
    ))
    
    # Add coordinate axes
    axis_length = max(lx, 2*ly, lz) * 0.2
    
    fig.add_trace(go.Scatter3d(
        x=[0, axis_length], y=[0, 0], z=[0, 0],
        mode='lines', line=dict(color='red', width=2),
        name='X', hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, axis_length], z=[0, 0],
        mode='lines', line=dict(color='green', width=2),
        name='Y', hoverinfo='skip'
    ))
    
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
        height=500,
        margin=dict(l=0, r=0, b=0, t=30),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        hovermode=False
    )
    
    return fig

# Optimized export function
@st.cache_data(ttl=300, show_spinner=False)
def export_mesh_cached(mesh_data, format_info):
    """Export mesh with caching"""
    format_name = format_info["name"]
    extension = format_info["extension"]
    backend = format_info.get("backend", "meshio")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"two_slabs.{extension}"
        filepath = os.path.join(tmpdir, filename)
        
        try:
            # Handle UNV format with custom writer
            if extension == "unv":
                write_unv_format(mesh_data, filepath)
            
            # Handle MSH format with custom writer
            elif extension == "msh":
                write_msh_format(mesh_data, filepath)
            
            # Handle text format
            elif extension == "txt":
                with open(filepath, 'w') as f:
                    f.write(f"# Mesh Data - ParallelGroup Mesh Generator\n")
                    f.write(f"# Dimensions: {mesh_data['dimensions']}\n")
                    f.write(f"# Points: {len(mesh_data['points'])}\n")
                    f.write(f"# Backend: {mesh_data.get('backend', 'unknown')}\n")
                    
                    if 'tetra' in mesh_data.get('cells', {}):
                        f.write(f"# Tetrahedra: {len(mesh_data['cells']['tetra'])}\n")
                    elif 'hexahedron' in mesh_data.get('cells', {}):
                        f.write(f"# Hexahedra: {len(mesh_data['cells']['hexahedron'])}\n")
                    
                    f.write("\n# Physical Groups:\n")
                    for name, info in PHYSICAL_GROUPS.items():
                        f.write(f"# {name}: dim={info['dim']}, id={info['id']}\n")
                    
                    f.write("\n# Points (first 1000):\n")
                    for i, point in enumerate(mesh_data['points'][:1000], 1):
                        f.write(f"{i} {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
            
            # Handle other formats with meshio if available
            elif st.session_state.import_status["HAS_MESHIO"] and backend == "meshio":
                meshio_module = st.session_state.import_status["meshio_data"]["meshio"]
                
                # Prepare cells for meshio
                if 'tetra' in mesh_data.get('cells', {}) and len(mesh_data['cells']['tetra']) > 0:
                    cells = [("tetra", mesh_data['cells']['tetra'][:2000])]
                elif 'hexahedron' in mesh_data.get('cells', {}) and len(mesh_data['cells']['hexahedron']) > 0:
                    cells = [("hexahedron", mesh_data['cells']['hexahedron'][:1000])]
                else:
                    # Create simple tetrahedral mesh
                    if len(mesh_data['points']) >= 4:
                        cells = [("tetra", np.array([[0, 1, 2, 3]]))]
                    else:
                        cells = []
                
                # Create meshio mesh
                mesh = meshio_module.Mesh(
                    mesh_data['points'][:1000],  # Limit points
                    cells,
                    point_data={}
                )
                
                # Map format names
                format_map = {
                    "vtk": "vtk",
                    "vtu": "vtu",
                    "stl": "stl",
                    "xdmf": "xdmf",
                    "med": "med",
                    "bdf": "nastran",
                    "inp": "abaqus",
                    "cdb": "ansys",
                    "ply": "ply",
                    "obj": "obj",
                    "dat": "tecplot",
                    "off": "off",
                    "su2": "su2",
                    "exo": "exodus"
                }
                
                file_format = format_map.get(extension, "vtk")
                mesh.write(filepath, file_format=file_format)
            
            else:
                # Fallback to text format
                return export_mesh_cached(mesh_data, {
                    "name": "Simple Text (.txt)",
                    "extension": "txt",
                    "backend": "custom"
                })
            
            # Read file for download
            with open(filepath, 'rb') as f:
                return f.read(), filename
            
        except Exception as e:
            st.warning(f"Export to {format_name} failed: {str(e)[:100]}")
            # Fallback to text format
            return export_mesh_cached(mesh_data, {
                "name": "Simple Text (.txt)",
                "extension": "txt",
                "backend": "custom"
            })

def main():
    # Display backend status in sidebar
    with st.sidebar:
        st.header("🔄 System Status")
        
        status_cols = st.columns(4)
        with status_cols[0]:
            status = "✅" if st.session_state.import_status["HAS_MESHPY"] else "❌"
            st.metric("MeshPy", status)
        with status_cols[1]:
            status = "✅" if st.session_state.import_status["HAS_GMSH"] else "❌"
            st.metric("Gmsh", status)
        with status_cols[2]:
            status = "✅" if st.session_state.import_status["HAS_MESHIO"] else "⚠️"
            st.metric("MeshIO", status)
        with status_cols[3]:
            status = "✅" if st.session_state.import_status["HAS_PYVISTA"] else "⚠️"
            st.metric("PyVista", status)
        
        st.markdown("---")
        st.header("⚙️ PropertyParams")
        
        # Dimensions with number input boxes
        col1, col2, col3 = st.columns(3)
        with col1:
            lx = st.number_input("Length X (mm)", 
                               min_value=1.0, 
                               max_value=10000.0, 
                               value=200.0, 
                               step=10.0,
                               format="%.1f")
        with col2:
            ly = st.number_input("Width Y (mm)", 
                               min_value=1.0, 
                               max_value=5000.0, 
                               value=50.0, 
                               step=5.0,
                               format="%.1f")
        with col3:
            lz = st.number_input("Height Z (mm)", 
                               min_value=0.1, 
                               max_value=1000.0, 
                               value=2.0, 
                               step=0.5,
                               format="%.1f")
        
        st.markdown("---")
        st.subheader("🎯 Mesh Settings")
        
        # Mesh backend selection
        backend_options = ["Fast Hex Mesh", "Tetra Mesh (Gmsh)", "Tetra Mesh (MeshPy)"]
        mesh_backend = st.selectbox(
            "Mesh Generation Method",
            backend_options,
            index=0
        )
        
        if "Hex" in mesh_backend:
            resolution = st.slider("Resolution", 3, 15, 6, 
                                 help="Higher = finer mesh, but slower")
            max_volume = None
            mesh_size = None
        elif "Gmsh" in mesh_backend:
            if st.session_state.import_status["HAS_GMSH"]:
                mesh_size = st.slider("Mesh Size Factor", 0.1, 5.0, 1.0, 0.1)
                max_volume = None
                resolution = None
            else:
                st.warning("Gmsh not available, using hex mesh")
                mesh_backend = "Fast Hex Mesh"
                resolution = 6
                max_volume = None
                mesh_size = None
        else:  # MeshPy
            if st.session_state.import_status["HAS_MESHPY"]:
                max_volume = st.slider("Max Element Volume", 0.5, 50.0, 5.0, 0.5)
                mesh_size = None
                resolution = None
            else:
                st.warning("MeshPy not available, using hex mesh")
                mesh_backend = "Fast Hex Mesh"
                resolution = 6
                max_volume = None
                mesh_size = None
        
        st.markdown("---")
        st.subheader("🎨 Visualization")
        show_points = st.checkbox("Show Mesh Points", value=True)
        show_wireframe = st.checkbox("Show Wireframe", value=True)
        
        st.markdown("---")
        st.subheader("📤 Export Settings")
        
        # Export format selection with descriptions
        selected_format = st.selectbox(
            "Export Format",
            list(EXPORT_FORMATS.keys()),
            index=0,
            help="Select the mesh format for export"
        )
        
        # Show format description
        format_info = EXPORT_FORMATS[selected_format]
        st.caption(f"📝 {format_info['description']}")
        
        st.markdown("---")
        st.subheader("🔄 Cache Control")
        
        # Clear cache button
        if st.button("Clear All Cache", type="secondary", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.mesh_cache = {}
            st.session_state.fig_cache = {}
            st.session_state.current_mesh = None
            st.session_state.current_stats = None
            st.session_state.current_mesh_obj = None
            st.rerun()
        
        # Show cache info
        cache_size = len(st.session_state.mesh_cache)
        st.caption(f"Cache size: {cache_size} meshes")
    
    # Main content area
    st.subheader("Mesh Generation Control")
    
    # Generate button
    generate_col1, generate_col2 = st.columns([3, 1])
    with generate_col1:
        if st.button("🚀 Generate Mesh", type="primary", use_container_width=True):
            start_time = time.time()
            
            with st.spinner(f"Generating mesh with {mesh_backend}..."):
                # Generate mesh based on selected backend
                if "Gmsh" in mesh_backend and st.session_state.import_status["HAS_GMSH"]:
                    mesh_data = create_gmsh_mesh_cached(lx, ly, lz, mesh_size)
                    backend_used = "Gmsh"
                elif "MeshPy" in mesh_backend and st.session_state.import_status["HAS_MESHPY"]:
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
                st.session_state.last_generation_time = time.time() - start_time
                
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
                    "dimensions": mesh_data["dimensions"],
                    "generation_time": st.session_state.last_generation_time
                }
                
                # Cache the mesh
                cache_key = f"{lx}_{ly}_{lz}_{mesh_backend}_{resolution if resolution else max_volume}"
                st.session_state.mesh_cache[cache_key] = mesh_data
    
    # Display results if mesh exists in session state
    if st.session_state.current_mesh is not None:
        mesh_data = st.session_state.current_mesh
        stats = st.session_state.current_stats
        
        # Display statistics
        st.subheader("📊 Mesh Statistics")
        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
        with col_stats1:
            st.metric("Generation Backend", stats["backend"])
        with col_stats2:
            st.metric("Nodes", f"{stats['points']:,}")
        with col_stats3:
            st.metric(stats["cell_type"], f"{stats['cells']:,}")
        with col_stats4:
            if stats.get("generation_time"):
                st.metric("Generation Time", f"{stats['generation_time']:.2f}s")
        
        # Create two columns for layout
        col_viz, col_info = st.columns([3, 1])
        
        with col_viz:
            st.subheader("🎨 3D Visualization")
            fig = visualize_mesh_with_plotly_cached(mesh_data, show_points, show_wireframe)
            st.plotly_chart(fig, use_container_width=True, 
                          config={'displayModeBar': True})
        
        with col_info:
            st.subheader("📋 Mesh Information")
            
            # Display dimension info
            lx, ly, lz = stats["dimensions"]
            st.markdown(f"""
            **Dimensions:**
            - Length X: **{lx} mm**
            - Width Y: **{2*ly} mm** (total)
            - Height Z: **{lz} mm**
            
            **Mesh Properties:**
            - Element Type: {stats["cell_type"]}
            - Physical Groups: 13
            - Cache Status: ✅ Active
            """)
            
            with st.expander("Physical Groups Details"):
                for name, info in PHYSICAL_GROUPS.items():
                    if info['dim'] == 3:
                        st.markdown(f"🔹 **{name}** (Volume {info['id']})")
                    else:
                        st.markdown(f"📐 **{name}** (Face {info['id']})")
        
        # Export section
        st.subheader("📥 Export Mesh")
        
        # Create export button
        file_data, filename = export_mesh_cached(mesh_data, {
            "name": selected_format,
            "extension": EXPORT_FORMATS[selected_format]["extension"],
            "mime": EXPORT_FORMATS[selected_format]["mime"],
            "backend": EXPORT_FORMATS[selected_format]["backend"]
        })
        
        # Download button
        st.download_button(
            label=f"📥 Download as {selected_format}",
            data=file_data,
            file_name=filename,
            mime=EXPORT_FORMATS[selected_format]["mime"],
            use_container_width=True
        )
        
        # Format compatibility information
        with st.expander("🔧 Format Compatibility Information"):
            st.markdown(f"""
            **{selected_format} Compatibility:**
            
            {EXPORT_FORMATS[selected_format]['description']}
            
            **Supported Software:**
            """)
            
            # Software compatibility matrix
            compatibility = {
                "Gmsh MSH (.msh)": ["Code_Aster", "GetFEM", "FEniCS", "Gmsh", "Elmer"],
                "I-DEAS UNV (.unv)": ["ANSYS", "Abaqus", "Nastran", "I-DEAS", "Femap"],
                "VTK (.vtk)": ["ParaView", "VTK", "VisIt", "Mayavi"],
                "STL (.stl)": ["3D Printers", "Blender", "CAD software"],
                "MED (.med)": ["Code_Aster", "SALOME"],
                "Abaqus INP (.inp)": ["Abaqus"],
                "ANSYS CDB (.cdb)": ["ANSYS"],
                "Nastran BDF (.bdf)": ["Nastran", "Patran"]
            }
            
            selected_software = compatibility.get(selected_format, ["General purpose"])
            for software in selected_software:
                st.markdown(f"- ✅ {software}")
        
        # Performance tips
        with st.expander("💡 Performance & Optimization Tips"):
            st.markdown("""
            **For Streamlit Cloud Performance:**
            1. ✅ Use **Fast Hex Mesh** for quickest generation
            2. ✅ Keep resolution **6-8** for balanced performance
            3. ✅ Export as **MSH** or **VTK** for full compatibility
            4. ✅ Clear cache periodically to free memory
            
            **Mesh Quality Tips:**
            1. Aspect Ratio: Keep LZ > LY/10 for good elements
            2. For FEA: Use tetrahedral meshes with Gmsh
            3. For visualization: Hex meshes render faster
            
            **Note:** All operations are cached for 5 minutes.
            """)
    
    else:
        # Show welcome/instructions
        st.info("👆 Click 'Generate Mesh' to create your first mesh!")
        
        # Quick start guide
        with st.expander("🚀 Quick Start Guide"):
            st.markdown("""
            ### Step-by-Step Guide:
            
            1. **Set Dimensions:**
               - Enter X, Y, Z dimensions in millimeters
               - Example: 200×100×10 mm
            
            2. **Choose Mesh Type:**
               - **Fast Hex Mesh**: Quick, structured elements
               - **Tetra Mesh (Gmsh)**: Higher quality, slower
               - **Tetra Mesh (MeshPy)**: Alternative tetra generator
            
            3. **Adjust Settings:**
               - Resolution: 6-8 for hex meshes
               - Mesh Size: 1.0 for tetra meshes
               - Max Volume: 5.0 for tetra quality
            
            4. **Visualize:**
               - Toggle points/wireframe display
               - Rotate 3D view with mouse
            
            5. **Export:**
               - Choose from 16+ formats
               - MSH for FEM, UNV for commercial software
               - STL for 3D printing
            
            **Pro Tip:** Start with Hex Mesh and resolution 6 for fastest results.
            """)
        
        # Example configurations
        st.subheader("💡 Example Configurations")
        col_ex1, col_ex2, col_ex3 = st.columns(3)
        
        with col_ex1:
            if st.button("Thin Slab", use_container_width=True):
                st.session_state.lx_example = 200.0
                st.session_state.ly_example = 50.0
                st.session_state.lz_example = 2.0
                st.rerun()
        
        with col_ex2:
            if st.button("Medium Block", use_container_width=True):
                st.session_state.lx_example = 100.0
                st.session_state.ly_example = 100.0
                st.session_state.lz_example = 20.0
                st.rerun()
        
        with col_ex3:
            if st.button("Thick Wall", use_container_width=True):
                st.session_state.lx_example = 300.0
                st.session_state.ly_example = 30.0
                st.session_state.lz_example = 100.0
                st.rerun()
    
    # Footer with comprehensive info
    st.markdown("---")
    col_footer1, col_footer2, col_footer3 = st.columns(3)
    
    with col_footer1:
        st.caption("✅ Optimized for Streamlit Cloud")
    
    with col_footer2:
        st.caption("🔧 16+ Export Formats Supported")
    
    with col_footer3:
        st.caption("⚡ Intelligent Caching System")

if __name__ == "__main__":
    main()
