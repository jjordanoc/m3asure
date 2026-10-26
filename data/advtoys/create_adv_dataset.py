"""Blender rendering script for Toys4K adversarial dataset.

Renders specified views producing both RGB and ground-truth Normal maps
via the compositor.

Usage (called by build_adversarial_dataset.py):
    blender -b -P create_adv_dataset.py -- \
        --blend_path /path/to/object.blend \
        --output_dir /path/to/output \
        --class_name airplane \
        --object_uid airplane_001 \
        --azimuths 0,90,180,270 \
        --samples 128
"""

import argparse
import math
import os
import sys

import bpy
import numpy as np

ELEVATION = 30.0
CAMERA_DIST = 2.7
RESOLUTION = 512


# ── GPU setup ─────────────────────────────────────────────────────────────────

def enable_gpus(device_type: str = "CUDA") -> None:
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = "GPU"

    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.refresh_devices()

        available = [t[0] for t in prefs.get_device_types(bpy.context)]
        if device_type not in available:
            for fallback in ("OPTIX", "CUDA"):
                if fallback in available:
                    device_type = fallback
                    break

        prefs.compute_device_type = device_type
        for device in prefs.devices:
            device.use = device.type == device_type

    except Exception as e:
        print(f"GPU init failed ({e}), falling back to CPU.")


# ── Scene helpers ─────────────────────────────────────────────────────────────

def reset_scene() -> None:
    for obj in list(bpy.data.objects):
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat, do_unlink=True)
    for img in list(bpy.data.images):
        bpy.data.images.remove(img, do_unlink=True)


def load_blend_object(blend_path: str) -> bpy.types.Object:
    """Load a Toys4K .blend file and return the mesh object."""
    with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
        data_to.objects = [name for name in data_from.objects]

    for obj in data_to.objects:
        if obj is not None:
            bpy.context.scene.collection.objects.link(obj)

    mesh_objs = [o for o in bpy.data.objects if o.type == "MESH"]
    if not mesh_objs:
        raise RuntimeError(f"No mesh objects found in {blend_path}")
    obj = mesh_objs[0]

    obj.rotation_mode = "XYZ"
    obj.rotation_euler = (np.radians(-90), 0, 0)

    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.location = (0, 0, 0)
    bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

    # Rescale to fit in a unit cube
    verts = np.array([v.co for v in obj.data.vertices])
    if verts.size > 0:
        obj.scale = obj.scale * 0.45 / max(np.max(np.abs(verts)), 1e-6)
        bpy.ops.object.transform_apply(scale=True)

    return obj


def setup_camera() -> bpy.types.Object:
    cam = bpy.context.scene.objects["Camera"]
    cam.data.lens = 35
    cam.data.sensor_width = 32
    # We will set rotation explicitly, so no constraint needed
    for constraint in cam.constraints:
        cam.constraints.remove(constraint)
    return cam


def set_camera_position(cam: bpy.types.Object, azimuth_deg: float, elevation_deg: float, distance: float, front: str, up: str) -> None:
    import mathutils
    
    def parse_axis(s: str) -> mathutils.Vector:
        s = s.lower()
        sign = -1 if s.startswith('-') else 1
        c = s[-1]
        v = [0.0, 0.0, 0.0]
        if c == 'x': v[0] = sign
        elif c == 'y': v[1] = sign
        elif c == 'z': v[2] = sign
        else: raise ValueError(f"Invalid axis: {s}")
        return mathutils.Vector(v)

    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    
    forward_vec = parse_axis(front)
    up_vec = parse_axis(up)
    right_vec = up_vec.cross(forward_vec)
    
    pos = distance * (forward_vec * math.cos(el) * math.cos(az) + 
                      right_vec * math.cos(el) * math.sin(az) + 
                      up_vec * math.sin(el))
    cam.location = pos

    if pos.length < 1e-6:
        return

    # LookAt matrix
    z_axis = pos.normalized()  # local +Z points away from origin
    x_axis = up_vec.cross(z_axis)
    if x_axis.length < 1e-6:
        # fallback if looking straight down/up
        x_axis = mathutils.Vector((1, 0, 0)) if up_vec.z else mathutils.Vector((0, 0, 1))
    x_axis.normalize()
    y_axis = z_axis.cross(x_axis).normalized()

    mat = mathutils.Matrix([
        [x_axis.x, y_axis.x, z_axis.x],
        [x_axis.y, y_axis.y, z_axis.y],
        [x_axis.z, y_axis.z, z_axis.z]
    ])
    
    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion = mat.to_quaternion()


# ── Compositor setup ─────────────────────────────────────────────────────────

def setup_compositor(rgb_dir: str, normal_dir: str) -> None:
    """Configure compositor to output RGB on white and Normal on black."""
    scene = bpy.context.scene
    scene.use_nodes = True
    scene.view_settings.view_transform = 'Standard'  # Prevent AgX/Filmic beige wash-out
    bpy.context.view_layer.use_pass_normal = True

    tree = scene.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)

    rl = tree.nodes.new("CompositorNodeRLayers")

    # White background node
    white = tree.nodes.new("CompositorNodeRGB")
    white.outputs[0].default_value = (1, 1, 1, 1)

    # RGB: composite over white background
    alpha_over_rgb = tree.nodes.new("CompositorNodeAlphaOver")
    tree.links.new(white.outputs[0], alpha_over_rgb.inputs[1])
    tree.links.new(rl.outputs["Image"], alpha_over_rgb.inputs[2])

    rgb_out = tree.nodes.new("CompositorNodeOutputFile")
    rgb_out.base_path = rgb_dir
    rgb_out.format.file_format = "PNG"
    rgb_out.format.color_mode = "RGB"
    rgb_out.file_slots[0].path = "rgb_"
    tree.links.new(alpha_over_rgb.outputs[0], rgb_out.inputs[0])

    # Normal: Swap Y and Z Axes
    sep_node = tree.nodes.new('CompositorNodeSepRGBA')
    comb_node = tree.nodes.new('CompositorNodeCombRGBA')
    tree.links.new(rl.outputs['Normal'], sep_node.inputs[0])
    tree.links.new(sep_node.outputs[0], comb_node.inputs[1]) # X -> Red
    tree.links.new(sep_node.outputs[1], comb_node.inputs[2]) # Y -> Blue (Swap)
    tree.links.new(sep_node.outputs[2], comb_node.inputs[0]) # Z -> Green (Swap)

    # Normal: Map [-1, 1] to [0, 1]
    mul_node = tree.nodes.new('CompositorNodeMixRGB')
    mul_node.blend_type = 'MULTIPLY'
    mul_node.inputs[0].default_value = 1.0 
    mul_node.inputs[2].default_value = (0.5, 0.5, 0.5, 1.0) 

    add_node = tree.nodes.new('CompositorNodeMixRGB')
    add_node.blend_type = 'ADD'
    add_node.inputs[0].default_value = 1.0 
    add_node.inputs[2].default_value = (0.5, 0.5, 0.5, 1.0)

    tree.links.new(comb_node.outputs[0], mul_node.inputs[1])
    tree.links.new(mul_node.outputs[0], add_node.inputs[1])

    # Normal: inject alpha from the Image pass, then composite over black
    set_alpha = tree.nodes.new("CompositorNodeSetAlpha")
    set_alpha.mode = "APPLY"
    tree.links.new(add_node.outputs[0], set_alpha.inputs["Image"])
    tree.links.new(rl.outputs["Alpha"], set_alpha.inputs["Alpha"])

    # Black background for Normal maps
    black = tree.nodes.new("CompositorNodeRGB")
    black.outputs[0].default_value = (0, 0, 0, 1)

    alpha_over_normal = tree.nodes.new("CompositorNodeAlphaOver")
    tree.links.new(black.outputs[0], alpha_over_normal.inputs[1])
    tree.links.new(set_alpha.outputs[0], alpha_over_normal.inputs[2])

    # --- THE FIX: Anti-Gamma Node ---
    # Cancel out the sRGB encoding to preserve raw linear math
    gamma_node = tree.nodes.new("CompositorNodeGamma")
    gamma_node.inputs[1].default_value = 2.2 # The exact inverse of sRGB
    tree.links.new(alpha_over_normal.outputs[0], gamma_node.inputs[0])

    normal_out = tree.nodes.new("CompositorNodeOutputFile")
    normal_out.base_path = normal_dir
    normal_out.format.file_format = "PNG"
    normal_out.format.color_mode = "RGB"
    normal_out.file_slots[0].path = "normal_"
    
    # CRITICAL: Force output to Raw/Linear to prevent pastel washout
    normal_out.format.color_management = 'OVERRIDE'
    normal_out.format.linear_colorspace_settings.name = 'Raw'
    
    tree.links.new(gamma_node.outputs[0], normal_out.inputs[0])


# ── Render settings ──────────────────────────────────────────────────────────

def setup_render(samples: int = 128) -> None:
    scene = bpy.context.scene
    render = scene.render
    render.engine = "CYCLES"
    render.resolution_x = RESOLUTION
    render.resolution_y = RESOLUTION
    render.resolution_percentage = 100
    render.film_transparent = True
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    scene.cycles.samples = samples


def setup_lighting() -> None:
    bpy.ops.object.light_add(type="AREA")
    light = bpy.data.objects["Area"]
    light.data.energy = 3000
    light.location = (0, 0, 5)
    light.scale = (100, 100, 100)


# ── Main ──────────────────────────────────────────────────────────────────────

def render_object(
    blend_path: str,
    output_dir: str,
    class_name: str,
    object_uid: str,
    azimuths: list[int],
    samples: int = 128,
    front: str = "+y", 
    up: str = "+z",
) -> None:
    rgb_dir = os.path.join(output_dir, class_name, object_uid, "rgb")
    normal_dir = os.path.join(output_dir, class_name, object_uid, "normal")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)

    reset_scene()
    setup_render(samples)
    setup_lighting()
    load_blend_object(blend_path)
    cam = setup_camera()
    setup_compositor(rgb_dir, normal_dir)

    for azimuth in azimuths:
        set_camera_position(cam, azimuth, ELEVATION, CAMERA_DIST, front, up)
        bpy.context.view_layer.update()

        # frame_set drives the automatic file numbering in compositor output nodes
        bpy.context.scene.frame_set(azimuth)
        bpy.ops.render.render(write_still=False)

    # Rename compositor outputs from rgb_0090.png → rgb_90.png convention
    for subdir, prefix in [(rgb_dir, "rgb_"), (normal_dir, "normal_")]:
        _rename_compositor_outputs(subdir, prefix, azimuths)

    n = len(azimuths)
    print(f"Rendered {object_uid} ({class_name}): {n} view(s) × RGB + Normal @ {samples} spp")


def _rename_compositor_outputs(directory: str, prefix: str, azimuths: list[int]) -> None:
    """Rename compositor frame-numbered files to clean angle names.

    Compositor outputs files like 'rgb_0000.png', 'rgb_0090.png' etc.
    We rename to 'rgb_0.png', 'rgb_90.png', etc.
    """
    for azimuth in azimuths:
        # Compositor zero-pads frame numbers to 4 digits
        src = os.path.join(directory, f"{prefix}{azimuth:04d}.png")
        dst = os.path.join(directory, f"{prefix}{azimuth}.png")
        if os.path.exists(src):
            os.rename(src, dst)
        else:
            print(f"WARNING: expected {src} not found")


if __name__ == "__main__":
    enable_gpus("OPTIX")

    parser = argparse.ArgumentParser()
    parser.add_argument("--blend_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--class_name", type=str, required=True)
    parser.add_argument("--object_uid", type=str, required=True)
    parser.add_argument("--azimuths", type=str, default="0,90,180,270",
                        help="Comma-separated azimuths to render")
    parser.add_argument("--samples", type=int, default=128,
                        help="Cycles render samples (lower = faster but noisier)")
    parser.add_argument("--front", type=str, default="+y",
                        help="Axis indicating the front of the object")
    parser.add_argument("--up", type=str, default="+z",
                        help="Axis indicating the top of the object")

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)
    azimuths = [int(a) for a in args.azimuths.split(",")]

    render_object(args.blend_path, args.output_dir, args.class_name, args.object_uid,
                  azimuths, args.samples, args.front, args.up)