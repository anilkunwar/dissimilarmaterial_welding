# app.py
import streamlit as st
import numpy as np
import tempfile
import os
import plotly.graph_objects as go
import io
import time
import hashlib
import json
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache

# Set page config at the very top
st.set_page_config(
    page_title="ParallelGroup Advanced Mesh Generator",
    page_icon="📐",
    layout="wide"
)

# ============================================================================
# DATA CLASSES FOR HASHABLE DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)  # Frozen makes it immutable and hashable
class MeshParameters:
    """Hashable container for mesh generation parameters"""
    lx: float
    ly: float
    lz: float
    backend: str
    resolution: Optional[float] = None
    max_volume: Optional[float] = None
    mesh_size: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'lx': self.lx,
            'ly': self.ly,
            'lz': self.lz,
            'backend': self.backend,
            'resolution': self.resolution,
            'max_volume': self.max_volume,
            'mesh_size': self.mesh_size
        }
    
    def to_hash(self) -> str:
        """Create a hash string for caching"""
        return hashlib.md5(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

@dataclass(frozen=True)
class VisualizationParameters:
    """Hashable container for visualization parameters"""
    show_points: bool
    show_wireframe: bool
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'show_points': self.show_points,
            'show_wireframe': self.show_wireframe
        }
    
    def to_hash(self) -> str:
        """Create a hash string for caching"""
        return hashlib.md5(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

# ============================================================================
# INITIALIZE SESSION STATE WITH HASHABLE STRUCTURES
# ============================================================================

if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.import_status = {}
    st.session_state.current_mesh_params = None
    st.session_state.current_viz_params = None
    st.session_state.current_mesh_data = None
    st.session_state.current_stats = None
    st.session_state.mesh_cache = {}
    st.session_state.viz_cache = {}
    st.session_state.export_cache = {}
    st.session_state.last_generation_time = None
    st.session_state.mesh_generation_count = 0

# ============================================================================
# CACHE FOR HEAVY IMPORTS
# ============================================================================

@st.cache_resource(ttl=3600, show_spinner=False)
def lazy_import_meshpy():
    """Lazy load MeshPy with caching"""
    try:
        from meshpy.tet import MeshInfo, build
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

# ============================================================================
# CONSTANTS - HASHABLE DATA
# ============================================================================

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
    }
}

st.title("ParallelGroup Advanced Mesh Generator")
st.markdown("""
This app generates structured and unstructured meshes using multiple backends.
**Optimized for Streamlit Cloud with intelligent caching.**
- **Slab 1 (Solid_1front)**: `(0,0,0)` → `(lx, ly, lz)`
- **Slab 2 (Solid_2back)**:  `(0,ly,0)` → `(lx, 2·ly, lz)`
""")

# ============================================================================
# MESH GENERATION FUNCTIONS WITH HASHABLE PARAMETERS
# ============================================================================

@st.cache_data(ttl=300, show_spinner=False, max_entries=10)
def create_mesh_with_meshpy_cached(_params: MeshParameters) -> Optional[Dict]:
    """Create mesh using MeshPy with caching"""
    if not st.session_state.import_status["HAS_MESHPY"]:
        return None
    
    try:
        MeshInfo = st.session_state.import_status["meshpy_data"]["MeshInfo"]
        build_func = st.session_state.import_status["meshpy_data"]["build"]
        
        mesh_info = MeshInfo()
        
        # Define vertices
        vertices = [
            (0, 0, 0), (_params.lx, 0, 0), (_params.lx, _params.ly, 0), (0, _params.ly, 0),
            (0, 0, _params.lz), (_params.lx, 0, _params.lz), (_params.lx, _params.ly, _params.lz), (0, _params.ly, _params.lz),
            (0, _params.ly, 0), (_params.lx, _params.ly, 0), (_params.lx, 2*_params.ly, 0), (0, 2*_params.ly, 0),
            (0, _params.ly, _params.lz), (_params.lx, _params.ly, _params.lz), (_params.lx, 2*_params.ly, _params.lz), (0, 2*_params.ly, _params.lz)
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
        
        max_volume = _params.max_volume if _params.max_volume is not None else 5.0
        
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
            "facet_markers": mesh.facet_markers.tolist() if hasattr(mesh.facet_markers, 'tolist') else mesh.facet_markers,
            "dimensions": (_params.lx, _params.ly, _params.lz),
            "backend": "meshpy",
            "params_hash": _params.to_hash()
        }
        
        return mesh_data
        
    except Exception as e:
        st.error(f"MeshPy generation failed: {str(e)[:100]}")
        return None

@st.cache_data(ttl=300, show_spinner=False, max_entries=10)
def create_gmsh_mesh_cached(_params: MeshParameters) -> Optional[Dict]:
    """Create mesh using Gmsh with proper physical groups"""
    if not st.session_state.import_status["HAS_GMSH"]:
        return None
    
    try:
        gmsh_module = st.session_state.import_status["gmsh_data"]["gmsh"]
        
        # Initialize Gmsh
        gmsh_module.initialize()
        gmsh_module.model.add("TwoSlabs")
        
        # Set mesh size
        lc = min(_params.lx, _params.ly, _params.lz) / 10.0
        if _params.mesh_size is not None:
            lc *= _params.mesh_size
        
        # Create first slab
        box1 = gmsh_module.model.occ.addBox(0, 0, 0, _params.lx, _params.ly, _params.lz)
        
        # Create second slab
        box2 = gmsh_module.model.occ.addBox(0, _params.ly, 0, _params.lx, _params.ly, _params.lz)
        
        # Synchronize
        gmsh_module.model.occ.synchronize()
        
        # Add physical groups
        gmsh_module.model.addPhysicalGroup(3, [box1], 1)
        gmsh_module.model.setPhysicalName(3, 1, "Solid_1front")
        
        gmsh_module.model.addPhysicalGroup(3, [box2], 2)
        gmsh_module.model.setPhysicalName(3, 2, "Solid_2back")
        
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
                "points": node_coords.tolist(),
                "cells": {"tetra": tets.tolist()},
                "physical_groups": PHYSICAL_GROUPS,
                "dimensions": (_params.lx, _params.ly, _params.lz),
                "backend": "gmsh",
                "params_hash": _params.to_hash()
            }
            
            gmsh_module.finalize()
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

@st.cache_data(ttl=300, show_spinner=False, max_entries=10)
def create_fallback_mesh_cached(_params: MeshParameters) -> Dict:
    """Create fallback mesh with caching"""
    # Optimized mesh generation - limit complexity
    resolution = _params.resolution if _params.resolution is not None else 6
    div_x = min(20, max(2, int(resolution * _params.lx / max(_params.lx, _params.ly, _params.lz))))
    div_y = min(20, max(2, int(resolution * _params.ly / max(_params.lx, _params.ly, _params.lz))))
    div_z = min(10, max(2, int(resolution * _params.lz / max(_params.lx, _params.ly, _params.lz))))
    
    # Generate points more efficiently
    x = np.linspace(0, _params.lx, div_x + 1)
    y = np.linspace(0, 2*_params.ly, 2*div_y + 1)
    z = np.linspace(0, _params.lz, div_z + 1)
    
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
        "points": points.tolist(),
        "cells": {"hexahedron": cells_array.tolist()},
        "physical_groups": PHYSICAL_GROUPS,
        "dimensions": (_params.lx, _params.ly, _params.lz),
        "divisions": (div_x, div_y, div_z),
        "backend": "fallback",
        "params_hash": _params.to_hash()
    }
    
    return mesh_data

# ============================================================================
# HELPER FUNCTIONS FOR MESH CONVERSION
# ============================================================================

def convert_mesh_data_for_use(mesh_data: Dict) -> Dict:
    """Convert lists back to numpy arrays for use in visualization and export"""
    if mesh_data is None:
        return None
    
    converted = mesh_data.copy()
    
    # Convert points back to numpy array
    if isinstance(converted["points"], list):
        converted["points"] = np.array(converted["points"])
    
    # Convert cells back to numpy arrays
    if "cells" in converted:
        for cell_type, cells in converted["cells"].items():
            if isinstance(cells, list):
                converted["cells"][cell_type] = np.array(cells)
    
    return converted

# ============================================================================
# UNV AND MSH FORMAT WRITERS
# ============================================================================

def write_unv_format(mesh_data: Dict, filepath: str) -> None:
    """Write mesh in I-DEAS UNV format"""
    points = mesh_data["points"]
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    
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
            if not isinstance(tets, np.ndarray):
                tets = np.array(tets)
            for tet in tets[:1000]:  # Limit for performance
                f.write(f"{element_counter:10d}{11:10d}{2:10d}{1:10d}{1:10d}{0:10d}{0:10d}\n")
                f.write(f"{tet[0]+1:10d}{tet[1]+1:10d}{tet[2]+1:10d}{tet[3]+1:10d}{0:10d}{0:10d}{0:10d}{0:10d}\n")
                element_counter += 1
        
        # Write hexahedral elements if available
        elif "hexahedron" in mesh_data["cells"] and len(mesh_data["cells"]["hexahedron"]) > 0:
            hexes = mesh_data["cells"]["hexahedron"]
            if not isinstance(hexes, np.ndarray):
                hexes = np.array(hexes)
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

def write_msh_format(mesh_data: Dict, filepath: str) -> None:
    """Write mesh in Gmsh MSH format version 2.2"""
    points = mesh_data["points"]
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    
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
            if not isinstance(tets, np.ndarray):
                tets = np.array(tets)
            for tet in tets[:1000]:  # Limit for performance
                elements_list.append((element_counter, 4, 2, 1, 1, 
                                     tet[0]+1, tet[1]+1, tet[2]+1, tet[3]+1))
                element_counter += 1
        
        # Collect hexahedral elements
        elif "hexahedron" in mesh_data["cells"] and len(mesh_data["cells"]["hexahedron"]) > 0:
            hexes = mesh_data["cells"]["hexahedron"]
            if not isinstance(hexes, np.ndarray):
                hexes = np.array(hexes)
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

# ============================================================================
# VISUALIZATION FUNCTION WITH HASHABLE PARAMETERS
# ============================================================================

@st.cache_data(ttl=300, show_spinner=False, max_entries=10)
def create_visualization_figure(_mesh_params: MeshParameters, _viz_params: VisualizationParameters) -> go.Figure:
    """Create optimized 3D visualization with hashable parameters"""
    # Get mesh data from cache or generate
    mesh_data = get_or_generate_mesh(_mesh_params)
    if mesh_data is None:
        return go.Figure()
    
    mesh_data = convert_mesh_data_for_use(mesh_data)
    points = mesh_data["points"]
    lx, ly, lz = mesh_data["dimensions"]
    
    fig = go.Figure()
    
    # Add mesh points if requested
    if _viz_params.show_points:
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
    if _viz_params.show_wireframe:
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

# ============================================================================
# MESH GENERATION DISPATCHER
# ============================================================================

def get_or_generate_mesh(params: MeshParameters) -> Optional[Dict]:
    """Get mesh from cache or generate new mesh"""
    cache_key = params.to_hash()
    
    # Check if in session state cache
    if cache_key in st.session_state.mesh_cache:
        return st.session_state.mesh_cache[cache_key]
    
    # Generate mesh based on backend
    mesh_data = None
    
    if params.backend == "Gmsh" and st.session_state.import_status["HAS_GMSH"]:
        mesh_data = create_gmsh_mesh_cached(params)
    elif params.backend == "MeshPy" and st.session_state.import_status["HAS_MESHPY"]:
        mesh_data = create_mesh_with_meshpy_cached(params)
    else:
        mesh_data = create_fallback_mesh_cached(params)
    
    # Store in cache
    if mesh_data is not None:
        st.session_state.mesh_cache[cache_key] = mesh_data
    
    return mesh_data

# ============================================================================
# EXPORT FUNCTION WITH HASHABLE PARAMETERS
# ============================================================================

@st.cache_data(ttl=300, show_spinner=False, max_entries=10)
def export_mesh_data(_mesh_params: MeshParameters, format_key: str) -> Tuple[bytes, str]:
    """Export mesh data with caching"""
    format_info = EXPORT_FORMATS.get(format_key, EXPORT_FORMATS["Simple Text (.txt)"])
    
    # Get mesh data
    mesh_data = get_or_generate_mesh(_mesh_params)
    if mesh_data is None:
        return b"", "error.txt"
    
    mesh_data = convert_mesh_data_for_use(mesh_data)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"two_slabs.{format_info['extension']}"
        filepath = os.path.join(tmpdir, filename)
        
        try:
            # Handle UNV format with custom writer
            if format_info['extension'] == "unv":
                write_unv_format(mesh_data, filepath)
            
            # Handle MSH format with custom writer
            elif format_info['extension'] == "msh":
                write_msh_format(mesh_data, filepath)
            
            # Handle text format
            elif format_info['extension'] == "txt":
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
                    points_array = mesh_data['points']
                    if isinstance(points_array, list):
                        points_array = np.array(points_array)
                    for i, point in enumerate(points_array[:1000], 1):
                        f.write(f"{i} {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
            
            # Handle other formats with meshio if available
            elif st.session_state.import_status["HAS_MESHIO"] and format_info['backend'] == "meshio":
                meshio_module = st.session_state.import_status["meshio_data"]["meshio"]
                
                # Prepare cells for meshio
                cells = []
                if 'tetra' in mesh_data.get('cells', {}) and len(mesh_data['cells']['tetra']) > 0:
                    tets = mesh_data['cells']['tetra']
                    if isinstance(tets, list):
                        tets = np.array(tets)
                    cells = [("tetra", tets[:2000])]
                elif 'hexahedron' in mesh_data.get('cells', {}) and len(mesh_data['cells']['hexahedron']) > 0:
                    hexes = mesh_data['cells']['hexahedron']
                    if isinstance(hexes, list):
                        hexes = np.array(hexes)
                    cells = [("hexahedron", hexes[:1000])]
                
                # Create meshio mesh
                points_array = mesh_data['points']
                if isinstance(points_array, list):
                    points_array = np.array(points_array)
                
                mesh = meshio_module.Mesh(
                    points_array[:1000],  # Limit points
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
                    "dat": "tecplot"
                }
                
                file_format = format_map.get(format_info['extension'], "vtk")
                mesh.write(filepath, file_format=file_format)
            
            else:
                # Fallback to text format
                return export_mesh_data(_mesh_params, "Simple Text (.txt)")
            
            # Read file for download
            with open(filepath, 'rb') as f:
                return f.read(), filename
            
        except Exception as e:
            # Fallback to text format
            return export_mesh_data(_mesh_params, "Simple Text (.txt)")

# ============================================================================
# MAIN APPLICATION FUNCTION
# ============================================================================

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
        
        # Backend-specific parameters
        resolution = None
        max_volume = None
        mesh_size = None
        
        if "Hex" in mesh_backend:
            resolution = st.slider("Resolution", 3, 15, 6, 
                                 help="Higher = finer mesh, but slower")
        elif "Gmsh" in mesh_backend:
            if st.session_state.import_status["HAS_GMSH"]:
                mesh_size = st.slider("Mesh Size Factor", 0.1, 5.0, 1.0, 0.1)
            else:
                st.warning("Gmsh not available, using hex mesh")
                mesh_backend = "Fast Hex Mesh"
                resolution = 6
        else:  # MeshPy
            if st.session_state.import_status["HAS_MESHPY"]:
                max_volume = st.slider("Max Element Volume", 0.5, 50.0, 5.0, 0.5)
            else:
                st.warning("MeshPy not available, using hex mesh")
                mesh_backend = "Fast Hex Mesh"
                resolution = 6
        
        st.markdown("---")
        st.subheader("🎨 Visualization")
        show_points = st.checkbox("Show Mesh Points", value=True)
        show_wireframe = st.checkbox("Show Wireframe", value=True)
        
        st.markdown("---")
        st.subheader("📤 Export Settings")
        
        # Export format selection
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
        
        col_cache1, col_cache2 = st.columns(2)
        with col_cache1:
            if st.button("Clear Cache", type="secondary", use_container_width=True):
                st.cache_data.clear()
                st.success("Cache cleared!")
                
        with col_cache2:
            cache_size = len(st.session_state.mesh_cache)
            st.metric("Cache Size", cache_size)
        
        st.caption(f"Mesh generations: {st.session_state.mesh_generation_count}")
    
    # Main content area
    st.subheader("Mesh Generation Control")
    
    # Generate button
    if st.button("🚀 Generate Mesh", type="primary", use_container_width=True):
        start_time = time.time()
        
        # Create hashable parameters
        mesh_params = MeshParameters(
            lx=lx,
            ly=ly,
            lz=lz,
            backend=mesh_backend,
            resolution=resolution,
            max_volume=max_volume,
            mesh_size=mesh_size
        )
        
        viz_params = VisualizationParameters(
            show_points=show_points,
            show_wireframe=show_wireframe
        )
        
        with st.spinner(f"Generating mesh with {mesh_backend}..."):
            # Generate or retrieve mesh
            mesh_data = get_or_generate_mesh(mesh_params)
            
            if mesh_data is None:
                st.error("Failed to generate mesh. Please try a different configuration.")
            else:
                # Store in session state
                st.session_state.current_mesh_params = mesh_params
                st.session_state.current_viz_params = viz_params
                st.session_state.current_mesh_data = mesh_data
                st.session_state.last_generation_time = time.time() - start_time
                st.session_state.mesh_generation_count += 1
                
                # Convert for statistics
                mesh_data_use = convert_mesh_data_for_use(mesh_data)
                
                # Calculate statistics
                num_points = len(mesh_data_use["points"])
                if 'tetra' in mesh_data_use.get('cells', {}):
                    num_cells = len(mesh_data_use['cells']['tetra'])
                    cell_type = "Tetrahedra"
                elif 'hexahedron' in mesh_data_use.get('cells', {}):
                    num_cells = len(mesh_data_use['cells']['hexahedron'])
                    cell_type = "Hexahedra"
                else:
                    num_cells = 0
                    cell_type = "Unknown"
                
                st.session_state.current_stats = {
                    "backend": mesh_data.get('backend', 'unknown'),
                    "points": num_points,
                    "cells": num_cells,
                    "cell_type": cell_type,
                    "dimensions": mesh_data['dimensions'],
                    "generation_time": st.session_state.last_generation_time
                }
    
    # Display results if mesh exists in session state
    if st.session_state.current_mesh_params is not None:
        mesh_params = st.session_state.current_mesh_params
        viz_params = st.session_state.current_viz_params
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
            
            # Create visualization figure using cached function
            fig = create_visualization_figure(mesh_params, viz_params)
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
        file_data, filename = export_mesh_data(mesh_params, selected_format)
        
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
        
        # Debug information
        with st.expander("🐛 Debug Information"):
            st.markdown(f"""
            **Cache Information:**
            - Mesh Cache Entries: {len(st.session_state.mesh_cache)}
            - Current Parameters Hash: {mesh_params.to_hash()}
            - Mesh Backend Used: {stats['backend']}
            
            **System Information:**
            - Streamlit Version: {st.__version__}
            - NumPy Version: {np.__version__}
            - Plotly Version: {go.__version__}
            """)
            
            if st.button("Show Mesh Data Structure"):
                st.json({
                    "dimensions": stats["dimensions"],
                    "points_count": stats["points"],
                    "cells_count": stats["cells"],
                    "cell_type": stats["cell_type"]
                })
    
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
               - Choose from 12+ formats
               - MSH for FEM, UNV for commercial software
               - STL for 3D printing
            
            **Pro Tip:** Start with Hex Mesh and resolution 6 for fastest results.
            """)
        
        # Example configurations
        st.subheader("💡 Example Configurations")
        col_ex1, col_ex2, col_ex3 = st.columns(3)
        
        with col_ex1:
            if st.button("Thin Slab", use_container_width=True):
                st.rerun()
        
        with col_ex2:
            if st.button("Medium Block", use_container_width=True):
                st.rerun()
        
        with col_ex3:
            if st.button("Thick Wall", use_container_width=True):
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.caption("✅ Streamlit Cloud Optimized • 🔧 Hashable Data Structures • ⚡ Intelligent Caching")

if __name__ == "__main__":
    main()
