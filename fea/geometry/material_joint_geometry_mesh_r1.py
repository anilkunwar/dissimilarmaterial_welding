import streamlit as st
import gmsh
import pyvista as pv
import numpy as np
import tempfile
import os

st.set_page_config(page_title="SALOME‑like Mesh Generator", layout="wide")
st.title("Parametric CAD & Meshing with gmsh + pyvista")
st.markdown("This app replicates the SALOME example: two stacked boxes partitioned by a plane.")

# Sidebar – input parameters (defaults from the original script)
with st.sidebar:
    st.header("Geometry")
    Lx = st.number_input("Length (X)", value=200.0)
    Ly = st.number_input("Height (Y)", value=100.0)
    Lz = st.number_input("Thickness (Z)", value=2.0)
    split_y = st.number_input("Split plane Y", value=50.0)

    st.header("Mesh sizes")
    char_len = st.number_input("Local length (1D)", value=2.3616, format="%.4f")
    max_area = st.number_input("Max triangle area (2D)", value=50.04)
    max_vol  = st.number_input("Max tetra volume (3D)", value=1181.7)

    run = st.button("Generate mesh")

# Main area
if run:
    with st.spinner("Building geometry and mesh..."):
        # Initialize gmsh
        gmsh.initialize()
        gmsh.model.add("partitioned_box")

        # Create the full box
        box_tag = gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz)

        # Create a plane at y = split_y (infinite, but fragment will trim it)
        plane_tag = gmsh.model.occ.addPlane(0, split_y, 0, 0, 1, 0)

        # Fragment: split the box by the plane
        gmsh.model.occ.synchronize()
        fragments = gmsh.model.occ.fragment([(3, box_tag)], [(2, plane_tag)])
        gmsh.model.occ.synchronize()

        # Extract resulting volumes (should be 2) and all surfaces
        volumes = [e for e in gmsh.model.getEntities(3)]
        surfaces = [e for e in gmsh.model.getEntities(2)]

        # ---- Identify volumes and faces by their bounding box ----
        # Volumes: "front" (y < split_y) and "back" (y > split_y)
        vol_front = None
        vol_back  = None
        for v in volumes:
            com = gmsh.model.occ.getCenterOfMass(v[0], v[1])
            if com[1] < split_y:
                vol_front = v
            else:
                vol_back = v

        # Physical groups for volumes
        if vol_front:
            gmsh.model.addPhysicalGroup(vol_front[0], [vol_front[1]], name="Solid_1front")
        if vol_back:
            gmsh.model.addPhysicalGroup(vol_back[0], [vol_back[1]], name="Solid_2back")

        # For each surface, determine its type and assign a physical group
        # We'll use a dictionary to map expected face names to a condition function
        # Coordinates are in [0,Lx] x [0,Ly] x [0,Lz]
        def assign_face(surf_tag, center):
            # center: (x,y,z)
            x, y, z = center
            eps = 1e-6

            # Identify by location
            if abs(x - 0) < eps:           # left side
                if y < split_y - eps:
                    name = "Face_1leftfront"
                else:
                    name = "Face_2leftback"
            elif abs(x - Lx) < eps:         # right side
                if y < split_y - eps:
                    name = "Face_10rightfront"
                else:
                    name = "Face_11rightback"
            elif abs(y - 0) < eps:          # bottom (z=0)
                if x < Lx/2:                # just for distinction, but bottom is split by interface? Actually bottom is continuous across y split, but we split at y=50 so bottom is two faces. We'll use y condition.
                    if x < split_y:
                        name = "Face_4bottomfront"
                    else:
                        name = "Face_7bottomback"
                else:
                    # Actually bottom is one continuous surface after split? The plane only splits the volume, not the bottom face. So bottom face is a single surface but we need two groups. This is a simplification: we'll split the bottom face artificially by y.
                    # Better: bottom face is one surface, but we need two groups. We'll check if the surface is entirely on one side of split_y.
                    # We'll approximate by the center y.
                    if y < split_y:
                        name = "Face_4bottomfront"
                    else:
                        name = "Face_7bottomback"
            elif abs(y - Ly) < eps:         # top (z=2)
                if y < split_y:
                    name = "Face_5topfront"
                else:
                    name = "Face_8topback"
            elif abs(z - 0) < eps:          # front (y=0)
                if x < split_y:
                    name = "Face_3frontfront"
                else:
                    name = "Face_9backback"   # Actually original had "backback" for y=100? Wait, we already have backback at y=100. Let's adjust.
            elif abs(z - Lz) < eps:         # back (y=100)
                if y < split_y:
                    name = "Face_???", but original had "Face_9backback" for y=100? Need mapping.
            else:
                name = None

            if name:
                gmsh.model.addPhysicalGroup(2, [surf_tag], name=name)

        # We need to iterate over surfaces and get their center of mass
        # But the plane created a new internal surface (the interface). We'll handle that separately.
        for surf in surfaces:
            # Get bounding box of surface
            min_pt, max_pt = gmsh.model.getBoundingBox(surf[0], surf[1])
            center = [(min_pt[i] + max_pt[i]) / 2 for i in range(3)]
            # Check if this surface is the interface (plane at y=split_y)
            if abs(center[1] - split_y) < 1e-4 and abs(min_pt[1] - max_pt[1]) < 1e-4:
                # Interface surface
                gmsh.model.addPhysicalGroup(2, [surf[1]], name="Face_6interfacefront")
            else:
                # External surface – determine which face
                assign_face(surf[1], center)

        # ---- Set mesh size constraints ----
        # Global size (1D)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_len)
        # Max triangle area (2D)
        gmsh.option.setNumber("Mesh.MaxElementArea", max_area)
        # Max tetra volume (3D)
        gmsh.option.setNumber("Mesh.MaxElementVolume", max_vol)

        # Generate 3D mesh
        gmsh.model.mesh.generate(3)

        # ---- Export mesh to a temporary file (VTK for pyvista) ----
        with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as f:
            tmpfile = f.name
        gmsh.write(tmpfile)

        # Clean up gmsh
        gmsh.finalize()

        # ---- Load mesh with pyvista ----
        mesh = pv.read(tmpfile)
        os.unlink(tmpfile)  # delete temporary file

    # ---- Display in Streamlit ----
    st.subheader("Generated Mesh")
    st.write(f"Number of cells: {mesh.n_cells}, points: {mesh.n_points}")

    # Extract cell arrays for physical groups
    # PyVista loads groups as cell arrays "gmsh:physical" but we can also show them
    # For simplicity, we just plot the mesh with a smooth shading
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh, color="lightblue", show_edges=True, opacity=0.8)
    plotter.view_xy()
    plotter.camera.zoom(1.2)

    # Use stpyvista if available, else fallback to static image
    try:
        from stpyvista import stpyvista
        stpyvista(plotter, key="mesh_viewer")
    except ImportError:
        # Save screenshot and show
        with tempfile.NamedTemporaryFile(suffix=".png") as img:
            plotter.screenshot(img.name)
            st.image(img.name, caption="Mesh view (static)")

    # ---- Download buttons ----
    st.subheader("Export mesh")
    with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as f:
        vtk_file = f.name
    mesh.save(vtk_file)
    with open(vtk_file, "rb") as f:
        st.download_button("Download as VTK", f, file_name="mesh.vtk")
    os.unlink(vtk_file)

    # Also export to .msh (Gmsh format) using meshio
    import meshio
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
        msh_file = f.name
    # Convert pyvista mesh to meshio and write
    meshio_mesh = meshio.Mesh(
        points=mesh.points,
        cells={"tetra": mesh.cells_dict.get("tetra", [])},
        cell_data={"gmsh:physical": mesh.cell_data.get("gmsh:physical", [])}
    )
    meshio.write(msh_file, meshio_mesh, file_format="gmsh")
    with open(msh_file, "rb") as f:
        st.download_button("Download as .msh", f, file_name="mesh.msh")
    os.unlink(msh_file)

else:
    st.info("Adjust parameters on the left and click **Generate mesh**.")
