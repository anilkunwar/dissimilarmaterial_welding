import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Butt Weld Joint Parametric GUI", layout="wide")
st.title("🔧 Butt Weld Joint – Parametric Geometry & Mesh Generator (SALOME)")
st.markdown("Change any coordinate or mesh value below → see live preview + download updated SALOME script")

# ====================== INPUT SECTION ======================
st.sidebar.header("Geometry Parameters")
st.sidebar.markdown("**Vertex 1** is fixed at (0, 0, 0)")

v2x = st.sidebar.number_input("Vertex 2 - X (Length of main plate)", value=200.0, step=0.01, format="%.4f")
v2y = st.sidebar.number_input("Vertex 2 - Y (Width of main plate)", value=50.0, step=0.01, format="%.4f")
v2z = st.sidebar.number_input("Vertex 2 - Z (Height / Thickness)", value=2.0, step=0.01, format="%.4f")

v3x = st.sidebar.number_input("Vertex 3 - X", value=0.0, step=0.01, format="%.4f")
v3y = st.sidebar.number_input("Vertex 3 - Y", value=100.0, step=0.01, format="%.4f")
v3z = st.sidebar.number_input("Vertex 3 - Z", value=0.0, step=0.01, format="%.4f")

st.sidebar.header("Mesh Parameters (quantitative)")
local_length = st.sidebar.number_input("1D Linear Local Length", value=2.3616, step=0.0001, format="%.4f")
max_area = st.sidebar.number_input("2D Triangular Max Element Area", value=50.04, step=0.01, format="%.4f")
max_vol = st.sidebar.number_input("3D Max Element Volume", value=1181.7, step=0.1, format="%.4f")

# ====================== GEOMETRY PREVIEW ======================
def add_box_edges(fig, xmin, ymin, zmin, xmax, ymax, zmax, color, name):
    edges = [
        [[xmin,ymin,zmin],[xmax,ymin,zmin]], [[xmax,ymin,zmin],[xmax,ymax,zmin]],
        [[xmax,ymax,zmin],[xmin,ymax,zmin]], [[xmin,ymax,zmin],[xmin,ymin,zmin]],
        [[xmin,ymin,zmax],[xmax,ymin,zmax]], [[xmax,ymin,zmax],[xmax,ymax,zmax]],
        [[xmax,ymax,zmax],[xmin,ymax,zmax]], [[xmin,ymax,zmax],[xmin,ymin,zmax]],
        [[xmin,ymin,zmin],[xmin,ymin,zmax]], [[xmax,ymin,zmin],[xmax,ymin,zmax]],
        [[xmax,ymax,zmin],[xmax,ymax,zmax]], [[xmin,ymax,zmin],[xmin,ymax,zmax]],
    ]
    for e in edges:
        fig.add_trace(go.Scatter3d(
            x=[e[0][0], e[1][0]], y=[e[0][1], e[1][1]], z=[e[0][2], e[1][2]],
            mode='lines', line=dict(color=color, width=5), name=name, showlegend=False
        ))
    return fig

fig = go.Figure()
add_box_edges(fig, 0, 0, 0, v2x, v2y, v2z, "blue", "Box_1 (front)")
bx2_xmin, bx2_xmax = min(v2x, v3x), max(v2x, v3x)
bx2_ymin, bx2_ymax = min(v2y, v3y), max(v2y, v3y)
bx2_zmin, bx2_zmax = min(v2z, v3z), max(v2z, v3z)
add_box_edges(fig, bx2_xmin, bx2_ymin, bx2_zmin, bx2_xmax, bx2_ymax, bx2_zmax, "red", "Box_2 (back)")

fig.update_layout(
    title="Live 3D Geometry Preview (Butt Weld Joint)",
    scene=dict(xaxis_title='X (Length)', yaxis_title='Y (Width)', zaxis_title='Z (Height)'),
    height=600, margin=dict(l=0, r=0, b=0, t=40)
)

st.plotly_chart(fig, use_container_width=True)

# ====================== MEASUREMENTS ======================
st.subheader("📏 Geometry Measurements")
col1, col2 = st.columns(2)
with col1:
    st.metric("Box 1 (Solid_1front)", f"Length {v2x} × Width {v2y} × Height {v2z}")
with col2:
    st.metric("Box 2 (Solid_2back)", 
              f"X-span {bx2_xmax-bx2_xmin} × Y-span {bx2_ymax-bx2_ymin} × Z-span {bx2_zmax-bx2_zmin}")

st.subheader("🧱 Mesh Parameters (will be used)")
st.write(f"**1D Linear Local Length:** {local_length}")
st.write(f"**2D Triangular Max Element Area:** {max_area}")
st.write(f"**3D Max Element Volume:** {max_vol}")

# ====================== GENERATE UPDATED SCRIPT ======================
original_script = """#!/usr/bin/env python
###
### This file is generated automatically by SALOME v9.9.0 with dump python functionality
###
import sys
import salome
salome.salome_init()
import salome_notebook
notebook = salome_notebook.NoteBook()
sys.path.insert(0, r'/home/geometry')
###
### GEOM component
###
import GEOM
from salome.geom import geomBuilder
import math
import SALOMEDS
geompy = geomBuilder.New()
O = geompy.MakeVertex(0, 0, 0)
OX = geompy.MakeVectorDXDYDZ(1, 0, 0)
OY = geompy.MakeVectorDXDYDZ(0, 1, 0)
OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)
Vector_1 = geompy.MakeVectorDXDYDZ(250, 150, 20)
Vertex_1 = geompy.MakeVertex(0, 0, 0)
Vertex_2 = geompy.MakeVertex(200, 50, 2)
Vertex_3 = geompy.MakeVertex(0, 100, 0)
Box_1 = geompy.MakeBoxTwoPnt(Vertex_1, Vertex_2)
Box_2 = geompy.MakeBoxTwoPnt(Vertex_2, Vertex_3)
Fuse_1 = geompy.MakeFuseList([Box_1, Box_2], True, True)
Partition_1 = geompy.MakePartition([Fuse_1], [Box_1], [], [], geompy.ShapeType["SOLID"], 0, [], 0)
[Solid_1front,Solid_2back] = geompy.ExtractShapes(Partition_1, geompy.ShapeType["SOLID"], True)
[Face_1leftfront,Face_2leftback,Face_3frontfront,Face_4bottomfront,Face_5topfront,Face_6interfacefront,Face_7bottomback,Face_8topback,Face_9backback,Face_10rightfront,Face_11rightback] = geompy.ExtractShapes(Partition_1, geompy.ShapeType["FACE"], True)
[Solid_1front, Solid_2back, Face_1leftfront, Face_2leftback, Face_3frontfront, Face_4bottomfront, Face_5topfront, Face_6interfacefront, Face_7bottomback, Face_8topback, Face_9backback, Face_10rightfront, Face_11rightback] = geompy.GetExistingSubObjects(Partition_1, False)
geompy.addToStudy( O, 'O' )
geompy.addToStudy( OX, 'OX' )
geompy.addToStudy( OY, 'OY' )
geompy.addToStudy( OZ, 'OZ' )
geompy.addToStudy( Vector_1, 'Vector_1' )
geompy.addToStudy( Vertex_1, 'Vertex_1' )
geompy.addToStudy( Vertex_2, 'Vertex_2' )
geompy.addToStudy( Vertex_3, 'Vertex_3' )
geompy.addToStudy( Box_1, 'Box_1' )
geompy.addToStudy( Box_2, 'Box_2' )
geompy.addToStudy( Fuse_1, 'Fuse_1' )
geompy.addToStudy( Partition_1, 'Partition_1' )
geompy.addToStudyInFather( Partition_1, Solid_1front, 'Solid_1front' )
geompy.addToStudyInFather( Partition_1, Solid_2back, 'Solid_2back' )
geompy.addToStudyInFather( Partition_1, Face_1leftfront, 'Face_1leftfront' )
geompy.addToStudyInFather( Partition_1, Face_2leftback, 'Face_2leftback' )
geompy.addToStudyInFather( Partition_1, Face_3frontfront, 'Face_3frontfront' )
geompy.addToStudyInFather( Partition_1, Face_4bottomfront, 'Face_4bottomfront' )
geompy.addToStudyInFather( Partition_1, Face_5topfront, 'Face_5topfront' )
geompy.addToStudyInFather( Partition_1, Face_6interfacefront, 'Face_6interfacefront' )
geompy.addToStudyInFather( Partition_1, Face_7bottomback, 'Face_7bottomback' )
geompy.addToStudyInFather( Partition_1, Face_8topback, 'Face_8topback' )
geompy.addToStudyInFather( Partition_1, Face_9backback, 'Face_9backback' )
geompy.addToStudyInFather( Partition_1, Face_10rightfront, 'Face_10rightfront' )
geompy.addToStudyInFather( Partition_1, Face_11rightback, 'Face_11rightback' )
###
### SMESH component
###
import SMESH, SALOMEDS
from salome.smesh import smeshBuilder
smesh = smeshBuilder.New()
Mesh_1 = smesh.Mesh(Partition_1,'Mesh_1')
Regular_1D = Mesh_1.Segment()
Local_Length_1 = Regular_1D.LocalLength(2.3616,None,1e-07)
MEFISTO_2D = Mesh_1.Triangle(algo=smeshBuilder.MEFISTO)
Max_Element_Area_1 = MEFISTO_2D.MaxElementArea(50.04)
NETGEN_3D = Mesh_1.Tetrahedron()
Max_Element_Volume_1 = NETGEN_3D.MaxElementVolume(1181.7)
Solid_1front_1 = Mesh_1.GroupOnGeom(Solid_1front,'Solid_1front',SMESH.VOLUME)
Solid_2back_1 = Mesh_1.GroupOnGeom(Solid_2back,'Solid_2back',SMESH.VOLUME)
Face_1leftfront_1 = Mesh_1.GroupOnGeom(Face_1leftfront,'Face_1leftfront',SMESH.FACE)
Face_2leftback_1 = Mesh_1.GroupOnGeom(Face_2leftback,'Face_2leftback',SMESH.FACE)
Face_3frontfront_1 = Mesh_1.GroupOnGeom(Face_3frontfront,'Face_3frontfront',SMESH.FACE)
Face_4bottomfront_1 = Mesh_1.GroupOnGeom(Face_4bottomfront,'Face_4bottomfront',SMESH.FACE)
Face_5topfront_1 = Mesh_1.GroupOnGeom(Face_5topfront,'Face_5topfront',SMESH.FACE)
Face_6interfacefront_1 = Mesh_1.GroupOnGeom(Face_6interfacefront,'Face_6interfacefront',SMESH.FACE)
Face_7bottomback_1 = Mesh_1.GroupOnGeom(Face_7bottomback,'Face_7bottomback',SMESH.FACE)
Face_8topback_1 = Mesh_1.GroupOnGeom(Face_8topback,'Face_8topback',SMESH.FACE)
Face_9backback_1 = Mesh_1.GroupOnGeom(Face_9backback,'Face_9backback',SMESH.FACE)
Face_10rightfront_1 = Mesh_1.GroupOnGeom(Face_10rightfront,'Face_10rightfront',SMESH.FACE)
Face_11rightback_1 = Mesh_1.GroupOnGeom(Face_11rightback,'Face_11rightback',SMESH.FACE)
isDone = Mesh_1.Compute()
[ Solid_1front_1, Solid_2back_1, Face_1leftfront_1, Face_2leftback_1, Face_3frontfront_1, Face_4bottomfront_1, Face_5topfront_1, Face_6interfacefront_1, Face_7bottomback_1, Face_8topback_1, Face_9backback_1, Face_10rightfront_1, Face_11rightback_1 ] = Mesh_1.GetGroups()
## Set names of Mesh objects
smesh.SetName(Regular_1D.GetAlgorithm(), 'Regular_1D')
smesh.SetName(Face_8topback_1, 'Face_8topback')
smesh.SetName(Face_9backback_1, 'Face_9backback')
smesh.SetName(NETGEN_3D.GetAlgorithm(), 'NETGEN 3D')
smesh.SetName(MEFISTO_2D.GetAlgorithm(), 'MEFISTO_2D')
smesh.SetName(Max_Element_Area_1, 'Max. Element Area_1')
smesh.SetName(Local_Length_1, 'Local Length_1')
smesh.SetName(Max_Element_Volume_1, 'Max. Element Volume_1')
smesh.SetName(Face_1leftfront_1, 'Face_1leftfront')
smesh.SetName(Face_2leftback_1, 'Face_2leftback')
smesh.SetName(Face_3frontfront_1, 'Face_3frontfront')
smesh.SetName(Face_4bottomfront_1, 'Face_4bottomfront')
smesh.SetName(Face_5topfront_1, 'Face_5topfront')
smesh.SetName(Face_6interfacefront_1, 'Face_6interfacefront')
smesh.SetName(Face_7bottomback_1, 'Face_7bottomback')
smesh.SetName(Mesh_1.GetMesh(), 'Mesh_1')
smesh.SetName(Face_11rightback_1, 'Face_11rightback')
smesh.SetName(Face_10rightfront_1, 'Face_10rightfront')
smesh.SetName(Solid_2back_1, 'Solid_2back')
smesh.SetName(Solid_1front_1, 'Solid_1front')
if salome.sg.hasDesktop():
  salome.sg.updateObjBrowser()
"""

# Apply user values (exact string replacement – safe because numbers are unique)
updated_script = original_script.replace(
    "MakeVertex(200, 50, 2)", f"MakeVertex({v2x}, {v2y}, {v2z})"
).replace(
    "MakeVertex(0, 100, 0)", f"MakeVertex({v3x}, {v3y}, {v3z})"
).replace(
    "LocalLength(2.3616,None,1e-07)", f"LocalLength({local_length},None,1e-07)"
).replace(
    "MaxElementArea(50.04)", f"MaxElementArea({max_area})"
).replace(
    "MaxElementVolume(1181.7)", f"MaxElementVolume({max_vol})"
)

# ====================== DOWNLOAD ======================
st.download_button(
    label="⬇️ Download Updated SALOME .py Script",
    data=updated_script,
    file_name="butt_weld_joint_updated.py",
    mime="text/plain"
)

st.success("✅ Script is ready! Load it in SALOME (File → Load Script) to see the exact geometry + mesh with your new dimensions.")
st.info("The 3D preview above is a live wireframe approximation. The real interactive 3D view + full mesh is generated inside SALOME.")
