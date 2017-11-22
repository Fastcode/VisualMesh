#!/usr/local/bin/blender -P

import bpy
import math
import os
import random
import json

def setup_background():
    # Get some variables
    scene = bpy.context.scene
    world = scene.world
    world.use_nodes = True
    node_tree = world.node_tree

    # Create the field environment node and set the image to it
    texture = node_tree.nodes.new(type='ShaderNodeTexEnvironment')

    # Create a texture mapping node to rotate the environment
    mapping = node_tree.nodes.new(type='ShaderNodeMapping')
    coord = node_tree.nodes.new(type='ShaderNodeTexCoord')

    # Connect color out of the node to the background
    node_tree.links.new(coord.outputs['Generated'], mapping.inputs['Vector'])
    node_tree.links.new(mapping.outputs['Vector'], texture.inputs['Vector'])
    node_tree.links.new(texture.outputs['Color'], node_tree.nodes['Background'].inputs['Color'])

    return (texture, mapping.rotation)

def setup_shadowcatcher():
    # Create the mesh
    bpy.ops.mesh.primitive_plane_add()
    obj = bpy.data.objects['Plane']

    mat = bpy.data.materials.new('ShadowCatcherMaterial')
    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes

    diffuse = nodes['Diffuse BSDF']
    texture = node_tree.nodes.new(type='ShaderNodeTexEnvironment')
    node_tree.links.new(texture.outputs['Color'], diffuse.inputs['Color'])

    obj.data.materials.append(mat)

    # Set it as a shadowcatcher
    bpy.data.objects['Plane'].cycles.is_shadow_catcher = True

    return texture

def create_pbr_ball_material():

    roughness = 0.05

    mat = bpy.data.materials.new('PBRBall')
    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes

    output = nodes['Material Output']

    # Textures
    color_tex = nodes.new(type='ShaderNodeTexImage')
    color_tex.name = color_tex.label = 'ColorTexture'

    normal_tex = nodes.new(type='ShaderNodeTexImage')
    normal_tex.color_space = 'NONE'
    normal_tex.name = normal_tex.label = 'NormalTexture'

    normal_tex_inv = nodes.new(type='ShaderNodeInvert')
    normal_tex_inv.name = normal_tex_inv.label = 'NormalTextureInvert'

    normal_map = nodes.new(type='ShaderNodeNormalMap')
    normal_map.name = normal_map.label = 'NormalMap'

    # Shaders
    glossy = nodes.new(type='ShaderNodeBsdfGlossy')
    glossy.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    glossy.inputs['Roughness'].default_value = roughness
    glossy.name = glossy.label = 'Gloss'

    diffuse = nodes['Diffuse BSDF']
    diffuse.inputs['Roughness'].default_value = roughness
    diffuse.name = diffuse.label = 'Diffuse'

    mix = nodes.new(type='ShaderNodeMixShader')
    mix.name = mix.label = 'GlossDiffuseMix'

    # Fresnel group
    fresnel = nodes.new(type='ShaderNodeFresnel')
    fresnel.name = fresnel.label = 'Fresnel'

    fresnel_bump = nodes.new(type='ShaderNodeBump')
    fresnel_bump.name = fresnel_bump.label = 'FresnelBump'

    fresnel_geometry = nodes.new(type='ShaderNodeNewGeometry')
    fresnel_geometry.name = fresnel_geometry.label = 'FresnelGeometry'

    fresnel_mix = nodes.new(type='ShaderNodeMixRGB')
    fresnel_mix.inputs['Fac'].default_value = roughness
    fresnel_mix.name = fresnel_mix.label = 'FresnelMix'

    reflection_mix = nodes.new(type='ShaderNodeMixRGB')
    reflection_mix.inputs[1].default_value = (0.05, 0.05, 0.05, 0.05)
    reflection_mix.inputs[2].default_value = (1.0, 1.0, 1.0, 1.0)
    reflection_mix.name = reflection_mix.label = 'Reflectivity'

    # Link our final shaders
    node_tree.links.new(diffuse.outputs['BSDF'], mix.inputs[1])
    node_tree.links.new(glossy.outputs['BSDF'], mix.inputs[2])
    node_tree.links.new(mix.outputs['Shader'], output.inputs['Surface'])

    # Link our Fresnel group
    node_tree.links.new(fresnel_mix.outputs['Color'], fresnel.inputs['Normal'])
    node_tree.links.new(fresnel_bump.outputs['Normal'], fresnel_mix.inputs[1])
    node_tree.links.new(fresnel_geometry.outputs['Incoming'], fresnel_mix.inputs[2])
    node_tree.links.new(fresnel.outputs['Fac'], reflection_mix.inputs['Fac'])
    node_tree.links.new(reflection_mix.outputs['Color'], mix.inputs['Fac'])

    # Link our textures
    node_tree.links.new(color_tex.outputs['Color'], diffuse.inputs['Color'])

    node_tree.links.new(normal_tex.outputs['Color'], normal_tex_inv.inputs['Color'])
    node_tree.links.new(normal_tex_inv.outputs['Color'], normal_map.inputs['Color'])

    node_tree.links.new(normal_map.outputs['Normal'], diffuse.inputs['Normal'])
    node_tree.links.new(normal_map.outputs['Normal'], glossy.inputs['Normal'])
    node_tree.links.new(normal_map.outputs['Normal'], fresnel_bump.inputs['Normal'])

def load_ball(ball):
    path = ball[0]
    color = ball[1]
    normal = ball[2]
    coeff = ball[3]

    bpy.ops.import_scene.fbx(filepath=path)

    obj = bpy.context.selected_objects[0]
    obj.pass_index = 1
    mat = bpy.data.materials['PBRBall']

    mat.node_tree.nodes['ColorTexture'].image = color
    mat.node_tree.nodes['NormalTexture'].image = normal

    obj.data.materials.append(mat)

    return obj

#####################################################################################################################
#####################################################################################################################
#                                                   CODE HERE                                                       #
#####################################################################################################################
#####################################################################################################################

# Change directories so we are where this file is
script_dir = os.path.dirname(os.path.realpath(__file__))

# Delete the initial cube and lamp objects
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()
bpy.ops.object.select_by_type(type='LAMP')
bpy.ops.object.delete()

# # Change to the cycles renderer and setup some options
bpy.context.scene.render.engine = 'CYCLES'
scene = bpy.data.scenes['Scene']
scene.cycles.device = 'CPU'
scene.cycles.samples = 128

# Enable the object pass index so we can make our masks
scene.render.layers['RenderLayer'].use_pass_object_index = True
scene.use_nodes = True

# Set resolution, and render at that resolution
scene.render.resolution_x = 1280
scene.render.resolution_y = 1024
scene.render.resolution_percentage = 100

# Setup each of our objects
world_tex_node, world_rotation = setup_background()
shadow_tex_node = setup_shadowcatcher()
create_pbr_ball_material()

# Link up our output layers to file output nodes
nodes = scene.node_tree.nodes
render_layers = scene.node_tree.nodes['Render Layers']
indexob_file = nodes.new('CompositorNodeOutputFile')
image_file = nodes.new('CompositorNodeOutputFile')
indexob_file.base_path = 'output'
indexob_file.file_slots[0].path = 'stencil'
image_file.base_path = 'output'
image_file.file_slots[0].path = 'image'

scene.node_tree.links.new(render_layers.outputs['IndexOB'], indexob_file.inputs['Image'])
scene.node_tree.links.new(render_layers.outputs['Image'], image_file.inputs['Image'])

# Move the camera to the origin and lock it to pointing at the shadow catcher plane
bpy.data.objects['Camera'].location = (0,0,0)
bpy.data.objects['Camera'].constraints.new(type='DAMPED_TRACK')
bpy.data.objects['Camera'].constraints['Damped Track'].target = bpy.data.objects['Plane']
bpy.data.objects['Camera'].constraints['Damped Track'].track_axis = 'TRACK_NEGATIVE_Z'
bpy.data.objects['Camera'].constraints['Damped Track'].influence = 0.75

# Load all of our balls
balls = []
for bdir in os.listdir(os.path.join(script_dir, 'textures', 'ball')):
    path = os.path.join(script_dir, 'textures', 'ball', bdir)
    if os.path.isdir(path):
        color_f = os.path.join(path, '{}_color.png'.format(bdir))
        normal_f = os.path.join(path, '{}_normal.png'.format(bdir))
        coeff_f = os.path.join(path, '{}_coeff.png'.format(bdir))
        mesh_f = os.path.join(path, '{}_mesh.fbx'.format(bdir))
        balls.append((mesh_f,
                      bpy.data.images.load(color_f),
                      bpy.data.images.load(normal_f),
                      bpy.data.images.load(coeff_f)))

# Load each of our environment textures
environments = []
for f in os.listdir(os.path.join(script_dir, 'textures', 'field')):
    if f.endswith('.hdr'):
        img = bpy.data.images.load(os.path.join(script_dir, 'textures', 'field', f))
        with open(os.path.join(script_dir, 'textures', 'field', '{}.json'.format(f[:-4])), 'r') as f:
            meta = json.loads(f.read())
        environments.append((img, meta))

# Set some initial values for our environment
environment = random.choice(environments)
environment = environments[1]
world_tex_node.image = environment[0]
shadow_tex_node.image = environment[0]
world_rotation[0] = math.radians(environment[1]['roll'])
world_rotation[1] = math.radians(environment[1]['pitch'])
world_rotation[2] = math.radians(environment[1]['yaw'])

# THINGS TO CHANGE GO HERE
shadowcatcher = bpy.data.objects['Plane']
camera = bpy.data.objects['Camera']
camera_data = bpy.data.cameras['Camera']
background_strength = bpy.context.scene.world.node_tree.nodes['Background'].inputs['Strength']

ball_radius = 0.075
ball = load_ball(random.choice(balls))
ball.dimensions = (ball_radius * 2.0, ball_radius * 2.0, ball_radius * 2.0)

ball.location = (1, 1, ball_radius - environment[1]['height'])
shadowcatcher.location = (1, 1, -environment[1]['height'])
shadowcatcher.scale = (ball_radius * 10, ball_radius * 10, 1.0)
camera.rotation_euler = (math.radians(63.559), 0, math.radians(-34.338))

# If we are running in the background render stuff
if bpy.app.background:
    fno = 1
    while True:

        run_data = {'ball' : {}, 'lens': {}, 'camera': {}}

        # Get a random environment
        f = random.choice(environments)
        b = random.choice(balls)

        height = f[1]['height']
        roll = math.radians(f[1]['roll'])
        pitch = math.radians(f[1]['pitch'])
        yaw = math.radians(f[1]['yaw'])

        # Apply our random environment
        world_tex_node.image = f[0]
        shadow_tex_node.image = f[0]
        world_rotation[0] = roll
        world_rotation[1] = pitch
        world_rotation[2] = yaw

        # Remove the old ball and add the new one
        bpy.data.objects.remove(ball)
        ball = load_ball(b)

        run_data['background'] = f[0].name
        run_data['ball']['file'] = b[0]
        run_data['ball']['radius'] = ball_radius

        # Randomize environment strength from 0.5 to 3
        background_strength.default_value = random.uniform(0.5, 3.0)
        run_data['background_strength'] = background_strength.default_value

        # Randomize ball rotation
        ball_orientation = (random.uniform(0, math.pi * 2), random.uniform(0, math.pi * 2), random.uniform(0, math.pi * 2))
        run_data['ball']['orientation'] = ball_orientation
        ball.rotation_euler = ball_orientation

        # Randomize ball position within the field bounds
        rang = random.choice(f[1]['ranges'])
        ball_loc = (random.uniform(rang[0][0], rang[0][1]), random.uniform(rang[1][0], rang[1][1]))
        run_data['ball']['position'] = ball_loc
        ball.location = (ball_loc[0], ball_loc[1], ball_radius - height)
        ball.dimensions = (ball_radius * 2.0, ball_radius * 2.0, ball_radius * 2.0)
        shadowcatcher.location = (ball_loc[0], ball_loc[1], -height)

        # Fisheye camera
        if random.choice([True, False]):
            camera_data.type = 'PANO'
            camera_data.cycles.panorama_type = 'FISHEYE_EQUISOLID'
            camera_data.cycles.fisheye_fov = random.uniform(math.pi / 2.0, math.pi)

            run_data['lens']['type'] = 'FISHEYE'
            run_data['lens']['fov'] = camera_data.cycles.fisheye_fov
            run_data['lens']['focal_length'] = camera_data.cycles.fisheye_lens
            run_data['lens']['sensor_width'] = camera_data.sensor_width
            run_data['lens']['sensor_height'] = camera_data.sensor_height


        # Perspective camera
        else:
            camera_data.type = 'PERSP'
            camera_data.lens_unit = 'FOV'
            camera_data.angle = random.uniform(math.radians(50), math.radians(100))

            run_data['lens']['type'] = 'PERSPECTIVE'
            run_data['lens']['fov'] = camera_data.angle


        # Randomize camera rotation
        camera_rotation = (random.uniform(math.pi / 4, math.pi / 2), random.uniform(-0.1, +0.1), 0)
        camera.rotation_euler = camera_rotation

        run_data['camera']['rotation'] = camera_rotation

        # Render the image
        bpy.context.scene.frame_set(fno)
        bpy.ops.render.render(write_still=True)

        # Save the metadata
        with open(os.path.join('output', 'meta{:04d}.json'.format(fno)), 'w') as md:
            md.write(json.dumps(run_data, sort_keys=True, indent=4, separators=(',', ': ')))

        fno += 1
