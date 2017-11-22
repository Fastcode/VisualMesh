#!/usr/local/bin/blender -P

import bpy
import bmesh
import math
import struct
import os
from mathutils import Vector, Euler

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# Delete the initial cube
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()
bpy.ops.object.select_by_type(type='LAMP')
bpy.ops.object.delete()
bpy.ops.object.select_by_type(type='CAMERA')
bpy.ops.object.delete()

# # Change to the cycles renderer
bpy.context.scene.render.engine = 'CYCLES'

def createmesh(verts, faces, uvs, name, count, id, subname, colors, normal_flag, normals, location):

    scn = bpy.context.scene
    mesh = bpy.data.meshes.new("mesh" + str(count))
    mesh.from_pydata(verts, [], faces)

    for i in range(len(uvs)):
        uvtex = mesh.uv_textures.new(name='UVMap')
    # for i in range(len(colors)):
    #     coltex = mesh.vertex_colors.new(name='col' + str(i))

    bm = bmesh.new()
    bm.from_mesh(mesh)

    # Create UV MAPS
    for i in range(len(uvs)):
        uvlayer = bm.loops.layers.uv['UVMap']
        for f in bm.faces:
            for l in f.loops:
                l[uvlayer].uv.x = uvs[i][l.vert.index][0]
                l[uvlayer].uv.y = 1 - uvs[i][l.vert.index][1]

    # # Create VERTEX COLOR MAPS
    # for i in range(len(colors)):
    #     collayer = bm.loops.layers.color['col' + str(i)]
    #     for f in bm.faces:
    #         for l in f.loops:
    #             l[collayer].r = colors[i][l.vert.index][0]
    #             l[collayer].g = colors[i][l.vert.index][1]
    #             l[collayer].b = colors[i][l.vert.index][2]


    # Finish up, write the bmesh back to the mesh
    bm.to_mesh(mesh)
    bm.free()  # free and prevent further access

    ###NEW WAY TO PASS NORMALS TO MESH
    if normal_flag:
        mesh.normals_split_custom_set_from_vertices(normals)
        mesh.use_auto_smooth=True

    object = bpy.data.objects.new(
        name + '_' + str(id) + '_' + str(count), mesh)

    object.location = location
    object.rotation_euler=Euler((1.570796251296997, -0.0, 0.0), 'XYZ')

    # Transformation attributes inherited from bounding boxes
    # if not name=='stadium':
    # object.scale=Vector((0.1,0.1,0.1))

    bpy.context.scene.objects.link(object)

    return object.name

def read_chars(f):
    s = ""
    for i in range(0x20):
        c = struct.unpack('B',f.read(1))[0]
        c = chr(c)
        if c: s+=c
    return s


for dirpath, dnames, fnames in os.walk("textures/2017"):
    for f in fnames:
        if f.endswith('.f2b'):

            # Delete old stuff
            bpy.ops.object.select_by_type(type='MESH')
            bpy.ops.object.delete()

            path = os.path.join(dirpath, f)
            print(os.path.join(dirpath, f))

            with open(path, 'rb') as f:

                magic, size, vxsize, bufcount, meshcount = struct.unpack('<2I2HI', f.read(0x10))
                model_name = read_chars(f).replace('\x00', '')

                for m in range(meshcount):
                    print('Parsing mesh:',m, 'Offset: ',hex(f.tell()))
                    sem, off, num1, num2 = struct.unpack('<4I',f.read(0x10))
                    print(hex(sem),hex(off), num1,num2)
                    back=f.tell()
                    verts = []
                    uv1 = []
                    uv2 = []
                    uvs=[]
                    normals = []
                    tangents = []
                    bitangents = []
                    colors = []
                    faces = []

                    if sem == 0x6873654D:
                        print('Creating Mesh')
                        #Mesh information
                        f.seek(off)
                        bufinfo = struct.unpack('<16B',f.read(0x10))
                        #Fetch vertices
                        for i in range(num1):
                            for j in range(bufcount):
                                #Skip data here
                                if not bufinfo[j]:
                                    if (j==0 or j==3 or j==4 or j==5):
                                        f.read(0xC)
                                    elif (j==1 or j==2):
                                        f.read(0x8)
                                else:
                                    if j==0:
                                        verts.append(struct.unpack('<3f',f.read(0xC)))
                                    elif j==1:
                                        uv1.append(struct.unpack('<2f',f.read(0x8)))
                                    elif j==2:
                                        uv2.append(struct.unpack('<2f',f.read(0x8)))
                                    elif j==3:
                                        normals.append(struct.unpack('<3f',f.read(0xC)))
                                    elif j==4:
                                        tangents.append(struct.unpack('<3f',f.read(0xC)))
                                    elif j==5:
                                        bitangents.append(struct.unpack('<3f',f.read(0xC)))


                        #Fetch Indices
                        ilength = 2
                        typ = 'H'
                        if (num2 > 0xFFFF):
                            ilength = 4
                            typ = 'I'

                        for i in range(num2):
                            faces.append(struct.unpack('<3'+typ,f.read(3*ilength)))

                        #Assemble stuff
                        if (len(uv1)>0):
                            uvs.append(uv1)
                        if (len(uv2)>0):
                            uvs.append(uv2)
                        if (len(normals)>0):
                            colors.append(normals)
                        if (len(tangents)>0):
                            colors.append(tangents)
                        if (len(bitangents)>0):
                            colors.append(bitangents)
                        if (len(normals)>0):
                            normal_flag = True
                        else:
                            normal_flag = False

                        name = model_name + '_' + str(m)

                        createmesh(verts, faces, uvs, name, m, m, '', colors, normal_flag, normals, Vector((0,0,0)))

                        # Export the resulting scene
                        bpy.ops.export_scene.fbx(filepath='{}.fbx'.format(path[:-4]), version='BIN7400')
