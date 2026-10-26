import torch
import pyrender
import trimesh
import numpy as np
import torch.nn.functional as F
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
from pytorch3d.structures import Meshes
from typing import Sequence

from .camera_alignment import project_with_affine_camera


def vis_mesh_3d(verts, faces, landmarks=None, vertex_color=None):
    """可视化 FLAME mesh

    Args:
        verts (torch.Tensor): 顶点，形状为 (B, 5023, 3)
        faces (torch.Tensor, optional): 三角面，形状为 (9976, 3), 为 None 时使用 FLAME 模型自带的三角面. Defaults to None.
        landmarks (torch.Tensor, optional): 三维关键点，形状为 (B, 68, 3). Defaults to None.
        vertex_color (torch.Tensor, optional): 顶点颜色，形状为 (B, 5023, 4). Defaults to None.
    """
    # verts: (B, 5023, 3)
    # faces: (9976, 3)
    # landmarks: (B, 68, 3)
    # vertex_color: (B, 5023, 4), RGBA

    verts_np = verts.detach().cpu().numpy()
    faces = faces.detach().cpu().numpy()

    if landmarks is not None:
        landmarks_np = landmarks.detach().cpu().numpy()

    if vertex_color is not None:
        vertex_color_np = vertex_color.detach().cpu().numpy()

    for i in range(verts.shape[0]):
        vertices = verts_np[i]
        if vertex_color is None:
            vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        else:
            vertex_colors = vertex_color_np[i]
        tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors, process=False)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=False)

        scene = pyrender.Scene(bg_color=np.array([0.05, 0.05, 0.05, 1.0]), ambient_light=np.array([0.15, 0.15, 0.15]))
        scene.add(mesh)

        if landmarks is not None:
            joints = landmarks_np[i]
            sm = trimesh.creation.uv_sphere(radius=0.005)
            sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
            tfs = np.tile(np.eye(4), (len(joints), 1, 1))
            tfs[:, :3, 3] = joints
            joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(joints_pcl)

        extent = np.ptp(vertices, axis=0)
        mesh_size = max(float(np.max(extent)), 1e-4)
        # 相机位于模型正前方（正 z 方向），沿 -z 方向观察。
        camera_pose = np.eye(4, dtype=np.float32)
        camera_z = float(vertices[:, 2].max()) + mesh_size * 2.5
        # 默认正面视角：以模型包围盒中心为画面中心。
        bounds_min = vertices.min(axis=0)
        bounds_max = vertices.max(axis=0)
        center = (bounds_min + bounds_max) / 2.0
        # xmag、ymag 表示正交相机水平和竖直方向的半范围。
        half_size = 0.5 * max(float(extent[0]), float(extent[1]))
        half_size = max(half_size * 1.15, 1e-4)
        camera = pyrender.OrthographicCamera(xmag=half_size, ymag=half_size, znear=1e-4, zfar=max(10.0, mesh_size * 10.0))
        camera_pose[:3, 3] = np.array([center[0], center[1], camera_z], dtype=np.float32)
        scene.add(camera, pose=camera_pose)

        pyrender.Viewer(scene, viewport_size=(640, 640), use_perspective_cam=False, use_raymond_lighting=True)


def vis_mesh_2d(verts, faces, landmarks=None, vertex_color=None, image_size=256, margin=1.15, landmark_radius=None):
    """将一批 FLAME 网格和可选的三维关键点离屏渲染为 RGB 图像。

    Args:
        verts (torch.Tensor): 顶点，形状为 ``(B, N, 3)``。
        faces (torch.Tensor, optional): 三角面索引，形状为 ``(F, 3)``
            或 ``(B, F, 3)``。为 None 时使用 FLAME 自带的三角面。
        landmarks (torch.Tensor, optional): 三维关键点，形状为
            ``(B, L, 3)``。关键点会以红色小球绘制。
        vertex_color (torch.Tensor, optional): 顶点颜色，形状为
            ``(B, N, 3)`` 或 ``(B, N, 4)``。既支持 ``[0, 1]``，
            也支持 ``[0, 255]`` 的颜色范围。
        image_size (int | tuple[int, int]): 输出尺寸。整数表示正方形；
            二元组按 ``(height, width)`` 解释。
        margin (float): 网格边界之外保留的画面比例，必须大于 1。
        landmark_radius (float, optional): 关键点球半径。为 None 时根据
            网格尺寸自动确定。

    Returns:
        torch.Tensor: RGB 图像，形状为 ``(B, 3, H, W)``，类型为
        ``float32``，取值范围为 ``[0, 1]``，设备与 ``verts`` 相同。

    Note:
        pyrender 的离屏渲染不参与自动求导，返回图像不包含梯度。
    """
    if not torch.is_tensor(verts):
        raise TypeError("verts 必须是 torch.Tensor")
    if verts.ndim != 3 or verts.shape[-1] != 3:
        raise ValueError(f"verts 的形状必须为 (B, N, 3)，当前为 {tuple(verts.shape)}")
    if margin <= 1.0:
        raise ValueError(f"margin 必须大于 1，当前为 {margin}")

    if isinstance(image_size, int):
        height = width = image_size
    elif isinstance(image_size, (tuple, list)) and len(image_size) == 2:
        height, width = image_size
    else:
        raise TypeError("image_size 必须是正整数或 (height, width) 二元组")
    height, width = int(height), int(width)
    if height <= 0 or width <= 0:
        raise ValueError(f"image_size 必须为正数，当前为 {(height, width)}")

    def to_numpy(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def to_rgba(colors):
        colors = np.asarray(colors)
        if colors.ndim != 2 or colors.shape[1] not in (3, 4):
            raise ValueError("单个样本的 vertex_color 形状必须为 (N, 3) 或 (N, 4)")
        colors = colors.astype(np.float32, copy=False)
        if colors.size and colors.max() <= 1.0:
            colors = colors * 255.0
        if colors.shape[1] == 3:
            alpha = np.full((colors.shape[0], 1), 255.0, dtype=np.float32)
            colors = np.concatenate([colors, alpha], axis=1)
        return np.clip(colors, 0.0, 255.0).astype(np.uint8)

    output_device = verts.device
    vertices_batch = to_numpy(verts)
    landmarks_batch = to_numpy(landmarks)
    colors_batch = to_numpy(vertex_color)
    faces_batch = np.asarray(to_numpy(faces), dtype=np.int64)
    batch_size, vertex_count = vertices_batch.shape[:2]

    if faces_batch.ndim not in (2, 3) or faces_batch.shape[-1] != 3:
        raise ValueError(f"faces 的形状必须为 (F, 3) 或 (B, F, 3)，当前为 {faces_batch.shape}")
    if faces_batch.ndim == 3 and faces_batch.shape[0] != batch_size:
        raise ValueError("faces 与 verts 的 batch size 不一致")
    if landmarks_batch is not None and (
        landmarks_batch.ndim != 3 or landmarks_batch.shape[0] != batch_size or landmarks_batch.shape[-1] != 3
    ):
        raise ValueError("landmarks 的形状必须为 (B, L, 3)，且 B 与 verts 一致")
    if colors_batch is not None and (
        colors_batch.ndim != 3
        or colors_batch.shape[0] != batch_size
        or colors_batch.shape[1] != vertex_count
        or colors_batch.shape[-1] not in (3, 4)
    ):
        raise ValueError("vertex_color 的形状必须为 (B, N, 3/4)，且 B、N 与 verts 一致")

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    rendered_images = []
    try:
        for i in range(batch_size):
            vertices = vertices_batch[i]
            faces = faces_batch[i] if faces_batch.ndim == 3 else faces_batch
            if colors_batch is None:
                vertex_colors = np.tile(np.array([77, 77, 77, 255], dtype=np.uint8), (vertex_count, 1))
            else:
                vertex_colors = to_rgba(colors_batch[i])

            tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors, process=False)
            mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=False)
            scene = pyrender.Scene(bg_color=np.array([0.05, 0.05, 0.05, 1.0]), ambient_light=np.array([0.25, 0.25, 0.25]))
            scene.add(mesh)

            extent = np.ptp(vertices, axis=0)
            mesh_size = max(float(np.max(extent)), 1e-4)

            if landmarks_batch is not None:
                joints = landmarks_batch[i]
                radius = mesh_size * 0.01 if landmark_radius is None else float(landmark_radius)
                if radius <= 0:
                    raise ValueError("landmark_radius 必须大于 0")
                sphere = trimesh.creation.uv_sphere(radius=radius)
                sphere.visual.vertex_colors = np.array([230, 26, 26, 255], dtype=np.uint8)
                poses = np.tile(np.eye(4, dtype=np.float32), (len(joints), 1, 1))
                poses[:, :3, 3] = joints
                scene.add(pyrender.Mesh.from_trimesh(sphere, poses=poses, smooth=False))

            bounds_min = vertices.min(axis=0)
            bounds_max = vertices.max(axis=0)
            center = (bounds_min + bounds_max) / 2.0

            # 使用保持宽高比的正交相机，从 +z 方向沿 -z 观察 FLAME。
            half_width = max(float(extent[0]) * 0.5 * margin, 1e-4)
            half_height = max(float(extent[1]) * 0.5 * margin, 1e-4)
            aspect = width / height
            ymag = max(half_height, half_width / aspect)
            xmag = ymag * aspect
            camera = pyrender.OrthographicCamera(xmag=xmag, ymag=ymag, znear=1e-4, zfar=max(10.0, mesh_size * 10.0))
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3, 3] = np.array([center[0], center[1], float(bounds_max[2]) + mesh_size * 2.5], dtype=np.float32)
            scene.add(camera, pose=camera_pose)

            # 离屏渲染没有 Viewer 的 Raymond lighting，因此显式添加相机灯。
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
            scene.add(light, pose=camera_pose)

            color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            rendered_images.append(color[..., :3])
    finally:
        renderer.delete()

    images = torch.from_numpy(np.stack(rendered_images, axis=0).copy())
    images = images.permute(0, 3, 1, 2).to(dtype=torch.float32).div_(255.0)
    return images.to(output_device)


def render_shape(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    cam: torch.Tensor,
    image_size: int | tuple[int, int] = 224,
    mesh_color: Sequence[float] = (180 / 255, 180 / 255, 180 / 255),
    background_color: Sequence[float] = (0.0, 0.0, 0.0),
    supersample: int = 2,
) -> torch.Tensor:
    """
    将批量三维网格渲染为灰色 shape 图像。

    Args:
        vertices:
            三维顶点，形状为 (B, V, 3)。
        faces:
            三角面索引，形状为 (F, 3)，索引从 0 开始。
        cam:
            支持两种相机：

            - ``(B, 3)``：DECA/EMOCA/SPECTRE 风格弱透视相机 ``[s, tx, ty]``；
            - ``(B, 2, 4)``：由 ``fit_affine_camera`` 拟合的三维仿射相机。
        image_size:
            输出图像边长，或 ``(height, width)``。
        mesh_color:
            网格的 RGB 颜色，取值范围为 [0, 1]。
        background_color:
            背景的 RGB 颜色，取值范围为 [0, 1]。
        supersample:
            超采样倍数。设为 2 可获得更平滑的轮廓。

    Returns:
        shape_images:
            渲染图像，形状为 (B, 3, image_size, image_size)，
            数值范围为 [0, 1]。
    """
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"vertices 应为 (B, V, 3)，实际为 {tuple(vertices.shape)}")
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError(f"faces 应为 (F, 3)，实际为 {tuple(faces.shape)}")
    is_weak_perspective = cam.ndim == 2 and cam.shape[-1] == 3
    is_affine = cam.ndim == 3 and cam.shape[-2:] == (2, 4)
    if not is_weak_perspective and not is_affine:
        raise ValueError(f"cam 应为 (B, 3) 或 (B, 2, 4)，实际为 {tuple(cam.shape)}")
    if cam.shape[0] != vertices.shape[0]:
        raise ValueError("vertices 与 cam 的批量大小必须一致")
    if supersample < 1:
        raise ValueError("supersample 必须大于等于 1")

    batch_size = vertices.shape[0]
    device = vertices.device
    dtype = vertices.dtype
    if isinstance(image_size, int):
        output_height = output_width = image_size
    else:
        if len(image_size) != 2:
            raise ValueError("image_size 应为整数或 (height, width)")
        output_height, output_width = int(image_size[0]), int(image_size[1])
    if output_height <= 1 or output_width <= 1:
        raise ValueError("image_size 的高度和宽度都必须大于 1")
    render_size = (output_height * supersample, output_width * supersample)

    faces = faces.to(device=device, dtype=torch.long)
    batch_faces = faces.unsqueeze(0).expand(batch_size, -1, -1)

    if is_weak_perspective:
        # DECA 弱透视投影：x' = s * (x + tx), y' = s * (y + ty)。
        scale = cam[:, None, 0:1]
        translation = cam[:, None, 1:3]
        projected_xy = scale * (vertices[..., :2] + translation)
        projected_depth = scale * vertices[..., 2:3]
    else:
        # 完整三维仿射投影能利用 z 坐标修正头部姿态和残余相机误差。
        projected_xy = project_with_affine_camera(vertices, cam)
        camera_x_axis = cam[:, 0, :3]
        camera_y_axis = cam[:, 1, :3]
        view_direction = torch.linalg.cross(camera_x_axis, camera_y_axis, dim=-1)
        view_direction = F.normalize(view_direction, dim=-1, eps=1e-6)
        projected_depth = torch.einsum("bvc,bc->bv", vertices, view_direction).unsqueeze(-1)

    projected_vertices = torch.cat((projected_xy, projected_depth), dim=-1)

    # 从 FLAME/DECA 坐标系转换至图像渲染坐标系。
    transformed_vertices = projected_vertices * projected_vertices.new_tensor([1.0, -1.0, -1.0])
    transformed_vertices = transformed_vertices + transformed_vertices.new_tensor([0.0, 0.0, 10.0])

    # PyTorch3D 的 NDC 横、纵轴约定与 DECA rasterizer 相反。
    raster_vertices = transformed_vertices * transformed_vertices.new_tensor([-1.0, -1.0, 1.0])

    # 世界空间的平滑顶点法线，用于 Gouraud/Phong 风格平滑着色。
    world_meshes = Meshes(verts=vertices, faces=batch_faces)
    vertex_normals = world_meshes.verts_normals_padded()

    batch_indices = torch.arange(batch_size, device=device)[:, None, None]
    face_vertex_normals = vertex_normals[batch_indices, batch_faces]

    # 投影空间法线仅用于去除明显的背面区域，与 DECA 一致。
    transformed_meshes = Meshes(verts=transformed_vertices, faces=batch_faces)
    transformed_normals = transformed_meshes.verts_normals_padded()
    face_transformed_normals = transformed_normals[batch_indices, batch_faces]

    raster_meshes = Meshes(verts=raster_vertices.float(), faces=batch_faces)

    pix_to_face, _, barycentric, _ = rasterize_meshes(
        raster_meshes,
        image_size=render_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=None,
        max_faces_per_bin=None,
        perspective_correct=False,
        cull_backfaces=False,
    )

    # 把顶点法线以重心坐标插值到每个像素，产生平滑表面。
    normal_map = interpolate_face_attributes(pix_to_face, barycentric, face_vertex_normals.reshape(-1, 3, 3))[..., 0, :]
    normal_map = F.normalize(normal_map, dim=-1, eps=1e-6)

    transformed_normal_map = interpolate_face_attributes(pix_to_face, barycentric, face_transformed_normals.reshape(-1, 3, 3))[
        ..., 0, :
    ]

    visible_mask = pix_to_face[..., 0] >= 0
    front_mask = transformed_normal_map[..., 2] < 0.15
    alpha = (visible_mask & front_mask).to(dtype=dtype)

    # 与 DECA/SPECTRE render_shape 相同的五方向光源。
    light_directions = vertices.new_tensor(
        [[-1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [0.0, 0.0, 1.0]]
    )
    light_directions = F.normalize(light_directions, dim=-1)

    diffuse = torch.einsum("bhwc,lc->bhwl", normal_map, light_directions)
    diffuse = diffuse.clamp(min=0.0).mean(dim=-1) * 1.7

    mesh_color_tensor = torch.as_tensor(mesh_color, device=device, dtype=dtype).view(1, 1, 1, 3)

    background_tensor = torch.as_tensor(background_color, device=device, dtype=dtype).view(1, 1, 1, 3)

    shape_images = mesh_color_tensor * diffuse[..., None]
    shape_images = shape_images * alpha[..., None] + background_tensor * (1.0 - alpha[..., None])
    shape_images = shape_images.clamp(0.0, 1.0)
    shape_images = shape_images.permute(0, 3, 1, 2).contiguous()

    # 超采样后缩小，减少网格轮廓锯齿。
    if supersample > 1:
        shape_images = F.interpolate(
            shape_images,
            size=(output_height, output_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    return shape_images
