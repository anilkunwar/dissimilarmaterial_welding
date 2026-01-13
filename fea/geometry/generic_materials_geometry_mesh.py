# app.py

import streamlit as st
import gmsh
import numpy as np
import os
import tempfile

def create_slab_geometry(lx, ly, lz):
    """
    Create two slabs using Gmsh:
      - Solid_1front: from (0, 0, 0) to (lx, ly, lz)
      - Solid_2back:  from (0, ly, 0) to (lx, 2*ly, lz)

    This mimics your SALOME layout where Box_2 starts at y=ly.
    All faces are explicitly tagged with names matching your original.
    """
    gmsh.initialize()
    gmsh.model.add("TwoSlabs")

    # Parameters
    lc = min(lx, ly, lz) / 10.0  # mesh size (adjustable)

    # === Define Points for Solid_1front (front slab) ===
    # Bottom face (z=0)
    p1 = gmsh.model.occ.addPoint(0,     0,     0, lc)   # origin
    p2 = gmsh.model.occ.addPoint(lx,    0,     0, lc)
    p3 = gmsh.model.occ.addPoint(lx,    ly,    0, lc)
    p4 = gmsh.model.occ.addPoint(0,     ly,    0, lc)
    # Top face (z=lz)
    p5 = gmsh.model.occ.addPoint(0,     0,     lz, lc)
    p6 = gmsh.model.occ.addPoint(lx,    0,     lz, lc)
    p7 = gmsh.model.occ.addPoint(lx,    ly,    lz, lc)
    p8 = gmsh.model.occ.addPoint(0,     ly,    lz, lc)

    # Lines for front slab
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

    # Surfaces (counterclockwise when viewed from outside)
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

    # Solid 1
    solid1_surf = gmsh.model.occ.addSurfaceLoop([s_bottom1, s_top1, s_front1, s_back1, s_left1, s_right1])
    solid1 = gmsh.model.occ.addVolume([solid1_surf])

    # === Define Points for Solid_2back (back slab: y from ly to 2*ly) ===
    # Bottom face (z=0)
    p9  = gmsh.model.occ.addPoint(0,     ly,    0, lc)
    p10 = gmsh.model.occ.addPoint(lx,    ly,    0, lc)
    p11 = gmsh.model.occ.addPoint(lx,    2*ly,  0, lc)
    p12 = gmsh.model.occ.addPoint(0,     2*ly,  0, lc)
    # Top face (z=lz)
    p13 = gmsh.model.occ.addPoint(0,     ly,    lz, lc)
    p14 = gmsh.model.occ.addPoint(lx,    ly,    lz, lc)
    p15 = gmsh.model.occ.addPoint(lx,    2*ly,  lz, lc)
    p16 = gmsh.model.occ.addPoint(0,     2*ly,  lz, lc)

    # Reuse shared points/lines where possible? For clarity, we redefine.
    # But note: p3=p10, p4=p9, p7=p14, p8=p13 → we could reuse, but let's be explicit.

    # Lines for back slab
    l13 = gmsh.model.occ.addLine(p9,  p10)
    l14 = gmsh.model.occ.addLine(p10, p11)
    l15 = gmsh.model.occ.addLine(p11, p12)
    l16 = gmsh.model.occ.addLine(p12, p9)
    l17 = gmsh.model.occ.addLine(p13, p14)
    l18 = gmsh.model.occ.addLine(p14, p15)
    l19 = gmsh.model.occ.addLine(p15, p16)
    l20 = gmsh.model.occ.addLine(p16, p13)
    l21 = gmsh.model.occ.addLine(p9,  p13)
    l22 = gmsh.model.occ.addLine(p10, p14)
    l23 = gmsh.model.occ.addLine(p11, p15)
    l24 = gmsh.model.occ.addLine(p12, p16)

    # Surfaces for back slab
    bottom2 = gmsh.model.occ.addCurveLoop([l13, l14, l15, l16])
    s_bottom2 = gmsh.model.occ.addPlaneSurface([bottom2])

    top2 = gmsh.model.occ.addCurveLoop([l17, l18, l19, l20])
    s_top2 = gmsh.model.occ.addPlaneSurface([top2])

    front2 = gmsh.model.occ.addCurveLoop([l13, l22, -l17, -l21])  # this is the interface!
    s_front2 = gmsh.model.occ.addPlaneSurface([front2])

    back2 = gmsh.model.occ.addCurveLoop([l15, l24, -l19, -l23])
    s_back2 = gmsh.model.occ.addPlaneSurface([back2])

    left2 = gmsh.model.occ.addCurveLoop([l16, l24, -l20, -l21])
    s_left2 = gmsh.model.occ.addPlaneSurface([left2])

    right2 = gmsh.model.occ.addCurveLoop([l14, l23, -l18, -l22])
    s_right2 = gmsh.model.occ.addPlaneSurface([right2])

    # Solid 2
    solid2_surf = gmsh.model.occ.addSurfaceLoop([s_bottom2, s_top2, s_front2, s_back2, s_left2, s_right2])
    solid2 = gmsh.model.occ.addVolume([solid2_surf])

    # Synchronize
    gmsh.model.occ.synchronize()

    # === Assign Physical Groups (matching your SALOME names) ===

    # Solids
    gmsh.model.addPhysicalGroup(3, [solid1], name="Solid_1front")
    gmsh.model.addPhysicalGroup(3, [solid2], name="Solid_2back")

    # Faces for Solid_1front
    gmsh.model.addPhysicalGroup(2, [s_left1],   name="Face_1leftfront")
    gmsh.model.addPhysicalGroup(2, [s_left2],   name="Face_2leftback")      # left of back slab
    gmsh.model.addPhysicalGroup(2, [s_front1],  name="Face_3frontfront")
    gmsh.model.addPhysicalGroup(2, [s_bottom1], name="Face_4bottomfront")
    gmsh.model.addPhysicalGroup(2, [s_top1],    name="Face_5topfront")
    gmsh.model.addPhysicalGroup(2, [s_front2],  name="Face_6interfacefront")  # interface = front of back slab
    gmsh.model.addPhysicalGroup(2, [s_bottom2], name="Face_7bottomback")
    gmsh.model.addPhysicalGroup(2, [s_top2],    name="Face_8topback")
    gmsh.model.addPhysicalGroup(2, [s_back2],   name="Face_9backback")
    gmsh.model.addPhysicalGroup(2, [s_right1],  name="Face_10rightfront")
    gmsh.model.addPhysicalGroup(2, [s_right2],  name="Face_11rightback")

    # Meshing
    gmsh.option.setNumber("Mesh.Algorithm", 6)        # Frontal-Delaunay for 2D
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)      # Delaunay for 3D
    gmsh.option.setNumber("Mesh.MeshSizeFactor", 1.0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", lc / 2)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc * 2)

    gmsh.model.mesh.generate(3)

    # Optional: Optimize
    gmsh.model.mesh.optimize("Netgen")

def main():
    st.title("ParallelGroup Slab Geometry Generator (Gmsh)")
    st.markdown("""
    Generate a 3D mesh of two adjacent slabs with named physical groups.
    - **Slab 1 (front)**: `(0,0,0)` → `(lx, ly, lz)`
    - **Slab 2 (back)**:  `(0,ly,0)` → `(lx, 2·ly, lz)`
    
    Output: `.msh` file compatible with FEM solvers.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        lx = st.number_input("Length (lx)", min_value=1.0, value=200.0, step=10.0)
    with col2:
        ly = st.number_input("Width (ly)", min_value=1.0, value=50.0, step=5.0)
    with col3:
        lz = st.number_input("Height (lz)", min_value=0.1, value=2.0, step=0.5)

    if st.button("🚀 Generate Mesh"):
        with st.spinner("Generating geometry and mesh..."):
            try:
                # Use temporary directory
                with tempfile.TemporaryDirectory() as tmpdir:
                    msh_path = os.path.join(tmpdir, "two_slabs.msh")
                    
                    create_slab_geometry(lx, ly, lz)
                    gmsh.write(msh_path)
                    gmsh.finalize()

                    # Read file for download
                    with open(msh_path, "rb") as f:
                        st.download_button(
                            label="📥 Download .msh File",
                            data=f.read(),
                            file_name="two_slabs.msh",
                            mime="application/octet-stream"
                        )
                    
                    st.success("✅ Mesh generated successfully!")
                    st.info(f"Domain size: {lx} × {2*ly} × {lz}")
                    
            except Exception as e:
                st.error(f"❌ Error during mesh generation: {str(e)}")
                gmsh.finalize()  # Ensure cleanup

    st.markdown("---")
    st.caption("Powered by [Gmsh](https://gmsh.info) + Streamlit | Physical group names match original SALOME model")

if __name__ == "__main__":
    main()
