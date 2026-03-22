import os

# ------------------------------------------------------------------------------
# 🔥 CRITICAL: Set BEFORE importing gmsh / pyvista
# ------------------------------------------------------------------------------
os.environ["GMSH_NO_OPENGL"] = "1"
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["DISPLAY"] = ""

import streamlit as st
import gmsh
import pyvista as pv
import numpy as np
import tempfile
import meshio

st.set_page_config(page_title="SALOME-like Mesh Generator", layout="wide")
st.title("Parametric CAD & Meshing (SALOME Replacement)")

st.markdown(
    "This app builds a two-layer block, splits it by a horizontal plane, "
    "and generates a 3D tetrahedral mesh with physical groups."
)

# ------------------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("Geometry")
    Lx = st.number_input("Length (X)", value=200.0, step=10.0)
    Ly = st.number_input("Height (Y)", value=100.0, step=10.0)
    Lz = st.number_input("Thickness (Z)", value=2.0, step=1.0)
    split_y = st.number_input("Split plane Y", value=50.0, step=5.0)

    st.header("Mesh sizes")
    char_len = st.number_input("Local length (1D)", value=2.36, step=0.1)
    max_area = st.number_input("Max triangle area (2D)", value=50.0, step=10.0)
    max_vol  = st.number_input("Max tetra volume (3D)", value=1180.0, step=100.0)

    run = st.button("Generate mesh", type="primary")

# ------------------------------------------------------------------------------
# Main generation
# ------------------------------------------------------------------------------
if run:
    with st.spinner("Building geometry and mesh..."):

        gmsh.initialize()
        gmsh.model.add("partitioned_box")

        # Create geometry
        box_tag = gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz)
        plane_tag = gmsh.model.occ.addPlane(0, split_y, 0, 0, 1, 0)

        gmsh.model.occ.synchronize()
        gmsh.model.occ.fragment([(3, box_tag)], [(2, plane_tag)])
        gmsh.model.occ.synchronize()

        # Identify volumes
        volumes = gmsh.model.getEntities(3)
        vol_lower, vol_upper = None, None

        for v in volumes:
            com = gmsh.model.occ.getCenterOfMass(v[0], v[1])
            if com[1] < split_y:
                vol_lower = v
            else:
                vol_upper = v

        if vol_lower:
            gmsh.model.addPhysicalGroup(3, [vol_lower[1]], name="Solid_1front")
        if vol_upper:
            gmsh.model.addPhysicalGroup(3, [vol_upper[1]], name="Solid_2back")

        # Identify surfaces
        surfaces = gmsh.model.getEntities(2)

        for surf in surfaces:
            min_pt, max_pt = gmsh.model.getBoundingBox(surf[0], surf[1])
            xc = (min_pt[0] + max_pt[0]) / 2
            yc = (min_pt[1] + max_pt[1]) / 2
            zc = (min_pt[2] + max_pt[2]) / 2

            eps = 1e-4
            name = None

            if abs(yc - split_y) < eps:
                name = "Interface"
            elif abs(xc - 0) < eps:
                name = "Left"
            elif abs(xc - Lx) < eps:
                name = "Right"
            elif abs(yc - 0) < eps:
                name = "Front"
            elif abs(yc - Ly) < eps:
                name = "Back"
            elif abs(zc - 0) < eps:
                name = "Bottom"
            elif abs(zc - Lz) < eps:
                name = "Top"

            if name:
                gmsh.model.addPhysicalGroup(2, [surf[1]], name=name)

        # Mesh settings
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_len)
        gmsh.option.setNumber("Mesh.MaxElementVolume", max_vol)

        gmsh.model.mesh.generate(3)

        # Export
        vtk_file = tempfile.NamedTemporaryFile(suffix=".vtk", delete=False).name
        unv_file = tempfile.NamedTemporaryFile(suffix=".unv", delete=False).name

        gmsh.write(vtk_file)
        gmsh.write(unv_file)

        gmsh.finalize()

        # Load mesh
        mesh = pv.read(vtk_file)

    # ------------------------------------------------------------------------------
    # Visualization (SAFE HEADLESS)
    # ------------------------------------------------------------------------------
    st.subheader("Generated Mesh")
    st.write(f"Cells: {mesh.n_cells}, Points: {mesh.n_points}")

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh, show_edges=True)
    plotter.view_xy()

    img_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    plotter.screenshot(img_file)

    st.image(img_file, caption="Mesh Preview")

    # ------------------------------------------------------------------------------
    # Downloads
    # ------------------------------------------------------------------------------
    st.subheader("Download Mesh")

    with open(unv_file, "rb") as f:
        st.download_button("Download UNV", f, "mesh.unv")

    with open(vtk_file, "rb") as f:
        st.download_button("Download VTK", f, "mesh.vtk")

    # Convert to msh
    msh_file = tempfile.NamedTemporaryFile(suffix=".msh", delete=False).name

    cells = {"tetra": mesh.cells_dict.get("tetra", [])}
    meshio.write_points_cells(msh_file, mesh.points, cells)

    with open(msh_file, "rb") as f:
        st.download_button("Download MSH", f, "mesh.msh")

else:
    st.info("Click **Generate mesh** to start.")
