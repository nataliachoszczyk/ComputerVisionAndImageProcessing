import os
import time as timer

import moderngl
import numpy as np
import pandas as pd
import torch
from moderngl_window import geometry
from PIL import Image

from data import INPUTS, PARAM_COLUMNS, sample_scene, scene_row
from model import load_generator
from phong_window import PhongWindow
from utils.config import WINDOW_SIZE

IMAGE_VERTEX_SHADER = '''
#version 330 core
in vec3 in_position;
in vec2 in_texcoord_0;
out vec2 uv;
void main() {
    uv = in_texcoord_0;
    gl_Position = vec4(in_position, 1.0);
}
'''

IMAGE_FRAGMENT_SHADER = '''
#version 330 core
uniform sampler2D image;
in vec2 uv;
out vec4 f_color;
void main() {
    f_color = vec4(texture(image, uv).rgb, 1.0);
}
'''


class NeuralWindow(PhongWindow):
    """Show the same scene twice: the Phong image of the renderer (left) and the image of the network (right).

    With --output_path, the window saves the pairs as compare_NNNN.png and the scene parameters in params.csv.
    """
    window_size = (2 * WINDOW_SIZE[0], WINDOW_SIZE[1])
    aspect_ratio = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.generator, self.input_kind = load_generator(self.argv.generator)
        self.phong_fbo = self.ctx.framebuffer(color_attachments=[self.ctx.texture(WINDOW_SIZE, 3)],
                                              depth_attachment=self.ctx.depth_renderbuffer(WINDOW_SIZE))
        self.network_texture = self.ctx.texture(WINDOW_SIZE, 3)
        for texture in (self.phong_fbo.color_attachments[0], self.network_texture):
            texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.image_program = self.ctx.program(vertex_shader=IMAGE_VERTEX_SHADER, fragment_shader=IMAGE_FRAGMENT_SHADER)
        self.left_quad = geometry.quad_2d(size=(1.0, 2.0), pos=(-0.5, 0.0))
        self.right_quad = geometry.quad_2d(size=(1.0, 2.0), pos=(0.5, 0.0))
        self.scene_time = None
        self.network_seconds = []

    @classmethod
    def add_arguments(cls, parser):
        super().add_arguments(parser)
        parser.add_argument('--generator', type=str, required=True, help='Path to the generator weights (generator.pt)')
        parser.add_argument('--scene_seconds', type=float, default=0.0,
                            help='Time of one scene in the window. 0 gives a new scene in each frame')

    def new_scene(self):
        scene = sample_scene(self.rng)
        self.phong_fbo.use()
        self.phong_fbo.clear(0.0, 0.0, 0.0, 0.0)
        self.draw_scene(scene)
        phong = np.asarray(self.read_image(self.phong_fbo))

        row = pd.DataFrame([scene_row(self.frame, scene)], columns=['index', 'split', *PARAM_COLUMNS])
        start = timer.perf_counter()
        with torch.no_grad():
            network = self.generator(torch.from_numpy(INPUTS[self.input_kind](row)))[0]
        self.network_seconds.append(timer.perf_counter() - start)
        network = (network.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        # A texture has its first row at the bottom.
        self.network_texture.write(np.ascontiguousarray(network[::-1]).tobytes())

        if self.output_path:
            Image.fromarray(np.hstack([phong, network])).save(os.path.join(self.output_path, f'compare_{self.frame:04}.png'))
            self.rows.append(scene_row(self.frame, scene))
        self.frame += 1

    def on_render(self, time: float, frame_time: float):
        if self.scene_time is None or time - self.scene_time >= self.argv.scene_seconds:
            self.scene_time = time
            self.new_scene()

        self.wnd.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.phong_fbo.color_attachments[0].use(0)
        self.left_quad.render(self.image_program)
        self.network_texture.use(0)
        self.right_quad.render(self.image_program)

        if self.frame == self.argv.n_images:
            print(f'network: {1000 * np.mean(self.network_seconds[1:]):.1f} ms per image on the CPU')
            if self.output_path:
                self.save_params()
            self.wnd.close()
