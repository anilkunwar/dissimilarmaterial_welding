# app.py
import streamlit as st
import numpy as np
import os
import tempfile
import sys

# Try to import gmsh - will fail in cloud environments
HAS_GMSH = False
try:
    import gmsh
    HAS_GMSH = True
except Exception as e:
    st.warning(f"Gmsh not available (running in cloud?): {str(e)}")
    HAS_GMSH = False

def generate_fallback_mesh(lx, ly, lz):
    """
    Generate a simple structured hex mesh for two slabs when Gmsh is unavailable.
    Outputs Gmsh ASCII format (v2.2) with physical groups matching the original names.
    """
    # Calculate coordinates
    y_interface = ly
    y_max = 2 * ly
    
    # Node coordinates (12 nodes total)
    nodes = [
        [0, 0, 0],       # 1
        [lx, 0, 0],      # 2
        [lx, y_interface, 0],  # 3
        [0, y_interface, 0],   # 4
        [0, 0, lz],      # 5
        [lx, 0, lz],     # 6
        [lx, y_interface, lz], # 7
        [0, y_interface, lz],  # 8
        [0, y_max, 0],   # 9
        [lx, y_max, 0],  # 10
        [lx, y_max, lz], # 11
        [0, y_max, lz]   # 12
    ]
    
    # Element connectivity (2 hexahedrons)
    elements = [
        [1, 2, 3, 4, 5, 6, 7, 8],   # Solid_1front
        [4, 3, 10, 9, 8, 7, 11, 12] # Solid_2back
    ]
    
    # Physical group definitions (matching original SALOME names)
    physical_groups = {
        "Solid_1front": {"dim": 3, "entities": [1]},
        "Solid_2back": {"dim": 3, "entities": [2]},
        "Face_1leftfront": {"dim": 2, "nodes": [1, 4, 8, 5]},
        "Face_2leftback": {"dim": 2, "nodes": [4, 9, 12, 8]},
        "Face_3frontfront": {"dim": 2, "nodes": [1, 2, 6, 5]},
        "Face_4bottomfront": {"dim": 2, "nodes": [1, 2, 3, 4]},
        "Face_5topfront": {"dim": 2, "nodes": [5, 6, 7, 8]},
        "Face_6interfacefront": {"dim": 2, "nodes": [4, 3, 7, 8]},
        "Face_7bottomback": {"dim": 2, "nodes": [4, 3, 10, 9]},
        "Face_8topback": {"dim": 2, "nodes": [8, 7, 11, 12]},
        "Face_9backback": {"dim": 2, "nodes": [9, 10, 11, 12]},
        "Face_10rightfront": {"dim": 2, "nodes": [2, 3, 7, 6]},
        "Face_11rightback": {"dim": 2, "nodes": [3, 10, 11, 7]}
    }
    
    # Build Gmsh ASCII format (v2.2)
    output = []
    output.append("$MeshFormat")
    output.append("2.2 0 8")
    output.append("$EndMeshFormat")
    
    # Nodes section
    output.append("$Nodes")
    output.append(str(len(nodes)))
    for i, (x, y, z) in enumerate(nodes, 1):
        output.append(f"{i} {x} {y} {z}")
    output.append("$EndNodes")
    
    # Elements section
    output.append("$Elements")
    # Count: 2 volumes + 11 surfaces = 13 elements
    output.append("13")
    
    # Volume elements (hexahedrons = type 5)
    for i, elem in enumerate(elements, 1):
        # Format: elm-number elm-type number-of-tags <tags> node-indices
        output.append(f"{i} 5 2 {i} {i} " + " ".join(map(str, elem)))
    
    # Surface elements (quadrangles = type 3)
    for i, (name, group) in enumerate(physical_groups.items(), start=3):
        if group["dim"] == 2:
            # Create element ID for this surface
            elem_id = i
            # Physical group ID (we'll use the index)
            phys_id = i - 2
            output.append(f"{elem_id} 3 2 {phys_id} {phys_id} " + " ".join(map(str, group["nodes"])))
    
    output.append("$EndElements")
    
    # Physical names section
    output.append("$PhysicalNames")
    output.append(str(len(physical_groups)))
    for i, name in enumerate(physical_groups.keys(), 1):
        dim = physical_groups[name]["dim"]
        output.append(f"{dim} {i} \"{name}\"")
    output.append("$EndPhysicalNames")
    
    return "\n".join(output)

def create_geometry_with_gmsh(lx, ly, lz):
    """Full Gmsh implementation (used when available)"""
    gmsh.initialize()
    gmsh.model.add("TwoSlabs")
    
    # Create geometry
    box1 = gmsh.model.occ.addBox(0, 0, 0, lx, ly, lz)
    box2 = gmsh.model.occ.addBox(0, ly, 0, lx, ly, lz)
    gmsh.model.occ.synchronize()
    
    # Create physical groups (matching SALOME names)
    volumes = gmsh.model.getEntities(3)
    gmsh.model.addPhysicalGroup(3, [volumes[0][1]], name="Solid_1front")
    gmsh.model.addPhysicalGroup(3, [volumes[1][1]], name="Solid_2back")
    
    # Get boundary entities for faces
    all_faces = gmsh.model.getEntities(2)
    
    # Create physical groups for faces
    face_names = [
        "Face_1leftfront", "Face_2leftback", "Face_3frontfront",
        "Face_4bottomfront", "Face_5topfront", "Face_6interfacefront",
        "Face_7bottomback", "Face_8topback", "Face_9backback",
        "Face_10rightfront", "Face_11rightback"
    ]
    
    for i, name in enumerate(face_names, 1):
        if i-1 < len(all_faces):
            gmsh.model.addPhysicalGroup(2, [all_faces[i-1][1]], tag=i, name=name)
    
    # Mesh generation
    gmsh.option.setNumber("Mesh.MeshSizeFactor", 0.5)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")
    
    # Get mesh data
    return gmsh.write("")

def main():
    st.title("ParallelGroup Slab Geometry Generator")
    st.markdown("""
    Generate 3D mesh of two adjacent slabs with named physical groups.
    - **Slab 1 (front)**: `(0,0,0)` → `(lx, ly, lz)`
    - **Slab 2 (back)**:  `(0,ly,0)` → `(lx, 2·ly, lz)`
    
    Works in cloud environments (no Gmsh required)!
    """)
    
    with st.sidebar:
        st.header("Geometry Parameters")
        lx = st.number_input("Length (lx)", min_value=1.0, value=250.0, step=10.0)
        ly = st.number_input("Width (ly)", min_value=1.0, value=50.0, step=5.0)
        lz = st.number_input("Height (lz)", min_value=0.1, value=10.0, step=0.5)
        
        if HAS_GMSH:
            st.success("✅ Gmsh available - full meshing capabilities")
        else:
            st.warning("⚠️ Running in cloud mode - using simplified mesh")
    
    if st.button("🚀 Generate Mesh", type="primary"):
        with st.spinner("Generating geometry and mesh..."):
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    mesh_path = os.path.join(tmpdir, "two_slabs.msh")
                    
                    if HAS_GMSH:
                        # Use full Gmsh implementation
                        mesh_data = create_geometry_with_gmsh(lx, ly, lz)
                        with open(mesh_path, "w") as f:
                            f.write(mesh_data)
                    else:
                        # Use fallback pure-Python implementation
                        mesh_data = generate_fallback_mesh(lx, ly, lz)
                        with open(mesh_path, "w") as f:
                            f.write(mesh_data)
                    
                    # Provide download button
                    with open(mesh_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Gmsh Mesh File (.msh)",
                            data=f.read(),
                            file_name="two_slabs.msh",
                            mime="application/octet-stream",
                            help="Compatible with Code_Aster, CalculiX, and other FEM solvers"
                        )
                    
                    # Display mesh info
                    st.success("✅ Mesh generated successfully!")
                    st.info(f"""
                    **Mesh Details:**
                    - Domain size: {lx} × {2*ly} × {lz}
                    - Elements: {'2 hexahedrons (fallback)' if not HAS_GMSH else 'Tetrahedral (full Gmsh)'}
                    - Physical groups: 2 volumes + 11 faces
                    """)
                    
                    # Show preview of physical groups
                    with st.expander("📋 Physical Group Names (matching SALOME)"):
                        st.markdown("""
                        **Volumes:**
                        - `Solid_1front`
                        - `Solid_2back`
                        
                        **Faces:**
                        - `Face_1leftfront`, `Face_2leftback`
                        - `Face_3frontfront`, `Face_9backback`
                        - `Face_4bottomfront`, `Face_7bottomback`
                        - `Face_5topfront`, `Face_8topback`
                        - `Face_6interfacefront` (internal interface)
                        - `Face_10rightfront`, `Face_11rightback`
                        """)
            
            except Exception as e:
                st.error(f"❌ Error during mesh generation: {str(e)}")
                st.exception(e)

    st.markdown("---")
    st.caption("💡 **Tip**: For complex geometries, run locally with Gmsh installed. This cloud version provides a simplified but functional mesh.")

if __name__ == "__main__":
    main()
