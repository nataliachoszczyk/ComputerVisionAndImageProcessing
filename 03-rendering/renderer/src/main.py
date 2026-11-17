import sys
from collections import namedtuple
from enum import Enum
from pathlib import Path

import moderngl_window

RENDERER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RENDERER_DIR.parent))  # data.py of the project

from neural_window import NeuralWindow
from phong_window import PhongWindow

Task = namedtuple('Task', ['window_args', 'window_cls'])


class TaskType(Enum):
    @property
    def window_args(self):
        return self.value.window_args

    @property
    def window_cls(self):
        return self.value.window_cls

    PHONG = Task(
        [
            f"--shaders_dir_path={RENDERER_DIR / 'resources' / 'shaders' / 'phong'}",
            "--shader_name=phong",
            "--model_name=sphere.obj",
            f"--output_path={RENDERER_DIR / 'output'}",
        ],
        PhongWindow
    )
    NEURAL = Task(
        [
            f"--shaders_dir_path={RENDERER_DIR / 'resources' / 'shaders' / 'phong'}",
            "--shader_name=phong",
            "--model_name=sphere.obj",
        ],
        NeuralWindow
    )


if __name__ == '__main__':
    # With the weights of a generator, the window also shows the image of the network.
    task = TaskType.NEURAL if any(arg.startswith('--generator') for arg in sys.argv) else TaskType.PHONG
    # Arguments from the command line override the defaults, for example --output_path or --window headless.
    moderngl_window.run_window_config(task.window_cls, args=task.window_args + sys.argv[1:])
