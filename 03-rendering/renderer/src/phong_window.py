import csv
import os.path

import moderngl
import numpy as np
from PIL import Image
from pyrr import Matrix44

from base_window import BaseWindow
from data import CAMERA_POSITION, CAMERA_TARGET, CAMERA_UP, FOV_DEGREES, NEAR, PARAM_COLUMNS, sample_scene, scene_row


class PhongWindow(BaseWindow):
    # Without a display the window counts as hidden, and the default limit is 30 frames per second.
    hidden_window_framerate_limit = 0

    def __init__(self, **kwargs):
        super(PhongWindow, self).__init__(**kwargs)
        self.frame = 0
        self.rng = np.random.default_rng(self.argv.seed)
        self.rows = []

    @classmethod
    def add_arguments(cls, parser):
        super().add_arguments(parser)
        parser.add_argument('--n_images', type=int, default=0, help='Close the window after this number of images')
        parser.add_argument('--seed', type=int, default=0)

    def init_shaders_variables(self):
        self.model_view_projection = self.program["model_view_projection"]
        self.model_matrix = self.program["model_matrix"]
        self.material_diffuse = self.program["material_diffuse"]
        self.material_shininess = self.program["material_shininess"]
        self.light_position = self.program["light_position"]
        self.camera_position = self.program["camera_position"]

    def on_render(self, time: float, frame_time: float):
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        scene = sample_scene(self.rng)
        material_diffuse = scene["diffuse"] / 255.0

        model_matrix = Matrix44.from_translation(scene["object_position"])
        proj = Matrix44.perspective_projection(FOV_DEGREES, self.aspect_ratio, NEAR, 1000.0)
        lookat = Matrix44.look_at(CAMERA_POSITION, CAMERA_TARGET, CAMERA_UP)

        model_view_projection = proj * lookat * model_matrix

        self.model_view_projection.write(model_view_projection.astype('f4').tobytes())
        self.model_matrix.write(model_matrix.astype('f4').tobytes())
        self.material_diffuse.write(np.array(material_diffuse, dtype='f4').tobytes())
        self.material_shininess.write(np.array([scene["shininess"]], dtype='f4').tobytes())
        self.light_position.write(np.array(scene["light_position"], dtype='f4').tobytes())
        self.camera_position.write(np.array(CAMERA_POSITION, dtype='f4').tobytes())

        self.vao.render()
        if self.output_path:
            img = (
                Image.frombuffer('RGB', self.wnd.size, self.wnd.fbo.read(components=3))
                     .transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            )
            img.save(os.path.join(self.output_path, f'image_{self.frame:04}.png'))
            self.rows.append(scene_row(self.frame, scene))
        self.frame += 1

        if self.frame == self.argv.n_images:
            self.save_params()
            self.wnd.close()

    def save_params(self):
        with open(os.path.join(self.output_path, 'params.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'split', *PARAM_COLUMNS])
            writer.writerows(self.rows)
