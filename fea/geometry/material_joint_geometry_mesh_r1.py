import streamlit as st
import gmsh
import pyvista as pv
import numpy as np
import tempfile
import os
#os.environ["GMSH_NO_OPENGL"] = "1"
import meshio

st.set_page_config(page_title="SALOME‑like Mesh Generator", layout="wide")
st.title("Parametric CAD & Meshing (SALOME Replacement)")
st.markdown("This app builds a two‑layer block, splits it by a horizontal plane, and generates a 3D tetrahedral mesh with physical groups – exactly like the SALOME example.")

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
    char_len = st.number_input("Local length (1D)", value=2.3616, format="%.4f", step=0.1)
    max_area = st.number_input("Max triangle area (2D)", value=50.04, step=10.0)
    max_vol  = st.number_input("Max tetra volume (3D)", value=1181.7, step=100.0)

    run = st.button("Generate mesh", type="primary")

# ------------------------------------------------------------------------------
# Main generation
# ------------------------------------------------------------------------------
if run:
    with st.spinner("Building geometry and mesh..."):
        # Initialize gmsh
        gmsh.initialize()
        gmsh.model.add("partitioned_box")

        # Create the full box
        box_tag = gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz)

        # Create a plane at y = split_y (infinite, but fragment will trim it)
        plane_tag = gmsh.model.occ.addPlane(0, split_y, 0, 0, 1, 0)

        # Synchronise and fragment: split the box by the plane
        gmsh.model.occ.synchronize()
        fragments = gmsh.model.occ.fragment([(3, box_tag)], [(2, plane_tag)])
        gmsh.model.occ.synchronize()

        # --- Identify volumes ---
        volumes = [e for e in gmsh.model.getEntities(3)]
        vol_lower = None   # y < split_y
        vol_upper = None   # y > split_y
        for v in volumes:
            com = gmsh.model.occ.getCenterOfMass(v[0], v[1])
            if com[1] < split_y - 1e-6:
                vol_lower = v
            else:
                vol_upper = v

        # Physical groups for volumes
        if vol_lower:
            gmsh.model.addPhysicalGroup(vol_lower[0], [vol_lower[1]], name="Solid_1front")
        if vol_upper:
            gmsh.model.addPhysicalGroup(vol_upper[0], [vol_upper[1]], name="Solid_2back")

        # --- Identify faces and assign physical groups ---
        surfaces = [e for e in gmsh.model.getEntities(2)]

        for surf in surfaces:
            # Get bounding box and centroid
            min_pt, max_pt = gmsh.model.getBoundingBox(surf[0], surf[1])
            center = [(min_pt[i] + max_pt[i]) / 2 for i in range(3)]
            xc, yc, zc = center

            # Determine the type of face
            eps = 1e-4
            if abs(yc - split_y) < eps and (max_pt[1] - min_pt[1]) < eps:
                # Interface plane
                name = "Face_6interfacefront"
            elif abs(xc - 0) < eps:        # left side
                if yc < split_y:
                    name = "Face_1leftfront"
                else:
                    name = "Face_2leftback"
            elif abs(xc - Lx) < eps:       # right side
                if yc < split_y:
                    name = "Face_10rightfront"
                else:
                    name = "Face_11rightback"
            elif abs(yc - 0) < eps:        # front (y=0)
                name = "Face_3frontfront"
            elif abs(yc - Ly) < eps:       # back (y=Ly)
                name = "Face_9backback"
            elif abs(zc - 0) < eps:        # bottom
                if yc < split_y:
                    name = "Face_4bottomfront"
                else:
                    name = "Face_7bottomback"
            elif abs(zc - Lz) < eps:       # top
                if yc < split_y:
                    name = "Face_5topfront"
                else:
                    name = "Face_8topback"
            else:
                name = None

            if name:
                gmsh.model.addPhysicalGroup(2, [surf[1]], name=name)

        # --- Set mesh size constraints (matching SALOME parameters) ---
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_len)
        gmsh.option.setNumber("Mesh.MaxElementArea", max_area)
        gmsh.option.setNumber("Mesh.MaxElementVolume", max_vol)

        # Generate 3D mesh
        gmsh.model.mesh.generate(3)

        # --- Export to UNV (primary format) and VTK (for visualisation) ---
        with tempfile.NamedTemporaryFile(suffix=".unv", delete=False) as f_unv:
            unv_file = f_unv.name
        with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as f_vtk:
            vtk_file = f_vtk.name

        gmsh.write(unv_file)
        gmsh.write(vtk_file)

        # Clean up gmsh
        gmsh.finalize()

        # Load VTK mesh for pyvista display
        mesh = pv.read(vtk_file)

    # --------------------------------------------------------------------------
    # Display results
    # --------------------------------------------------------------------------
    st.subheader("Generated Mesh")
    st.write(f"Number of cells: {mesh.n_cells}, points: {mesh.n_points}")

    # Show mesh with pyvista (interactive if stpyvista is installed)
    plotter = pv.Plotter(off_screen=False)
    plotter.add_mesh(mesh, color="lightblue", show_edges=True, opacity=0.8)
    plotter.view_xy()
    plotter.camera.zoom(1.2)

    try:
        from stpyvista import stpyvista
        stpyvista(plotter, key="mesh_viewer")
    except ImportError:
        # Fallback: static image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as img:
            plotter.screenshot(img.name)
            st.image(img.name, caption="Mesh view (static)")
            os.unlink(img.name)

    # --------------------------------------------------------------------------
    # Download buttons
    # --------------------------------------------------------------------------
    st.subheader("Export mesh")

    # UNV
    with open(unv_file, "rb") as f:
        st.download_button("Download as .unv", f, file_name="mesh.unv")
    os.unlink(unv_file)

    # VTK
    with open(vtk_file, "rb") as f:
        st.download_button("Download as .vtk", f, file_name="mesh.vtk")
    os.unlink(vtk_file)

    # Also export as Gmsh .msh format (optional)
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f_msh:
        msh_file = f_msh.name
    # Use meshio to convert pyvista mesh to .msh
    cells = {"tetra": mesh.cells_dict.get("tetra", [])}
    meshio_mesh = meshio.Mesh(points=mesh.points, cells=cells,
                              cell_data={"gmsh:physical": mesh.cell_data.get("gmsh:physical", [])})
    meshio.write(msh_file, meshio_mesh, file_format="gmsh")
    with open(msh_file, "rb") as f:
        st.download_button("Download as .msh", f, file_name="mesh.msh")
    os.unlink(msh_file)

else:
    st.info("Adjust the parameters on the left and click **Generate mesh**.")
