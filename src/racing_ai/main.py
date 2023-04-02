##################################################
##################### IMPORTS ####################
##################################################

import pyglet
import numpy as np
import torch

from racing_ai.car import Car
from racing_ai.track import Track
from racing_ai.ml.train import DQN

FPS = 30


##################################################
##################### CLASSES ####################
##################################################


class Window(pyglet.window.Window):
    def __init__(self):
        super(Window, self).__init__(width=1280, height=720)
        self.set_visible()

        self.track = Track()
        self.track.load("assets/track_1.txt")

        self.drag = pyglet.shapes.Line(0, 0, 0, 0, 3, color=(250, 30, 30))
        self.drag.opacity = 250

        self.car = Car(400, 200, 0.95, self.track)

        self.model_enabled = False
        self.model = DQN(len(self.car.sensors) + 1, 5)
        self.model.load_state_dict(torch.load("target_net.pth"))

    def update_from_model(self, dt):
        state = np.append(self.car.sensors, [self.car.speed], axis=0)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = self.model(state_tensor)
        action = action_tensor.detach().numpy()
        self.car.move(np.argmax(action), dt)

    def on_draw(self):
        self.clear()

        if self.track.track_visible:
            self.track.road_batch.draw()
            self.track.wall_batch.draw()
            self.track.gate_batch.draw()

        if self.track.rays_visible:
            self.track.temp_batch.draw()

        self.car.draw()
        self.car.vector.draw()
        self.drag.draw()

        pyglet.text.Label(
            f"Score: {str(self.car.target_reward_gate)}",
            color=(0, 0, 0, 255),
            font_name="Arial",
            font_size=16,
            x=1150,
            y=50,
            anchor_x="center",
            anchor_y="center",
        ).draw()

        pyglet.text.Label(
            f"Model Enabled: {str(self.model_enabled)}",
            color=(0, 0, 0, 255),
            font_name="Arial",
            font_size=16,
            x=1150,
            y=25,
            anchor_x="center",
            anchor_y="center",
        ).draw()

    def on_mouse_press(self, x, y, button, modifiers):
        self.track.create_track(x, y)
        self.drag.position = (x, y)

    def on_mouse_release(self, x, y, button, modifiers):
        self.track.create_track(x, y)
        self.drag.position = (0, 0)

    def on_mouse_drag(self, x, y, dx, dy, button, modifiers):
        self.drag.x2 = x
        self.drag.y2 = y

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.UP:
            pyglet.clock.schedule_interval(self.car.forward, 1 / FPS)
        elif symbol == pyglet.window.key.DOWN:
            pyglet.clock.schedule_interval(self.car.backward, 1 / FPS)
        elif symbol == pyglet.window.key.LEFT:
            pyglet.clock.schedule_interval(self.car.left, 1 / FPS)
        elif symbol == pyglet.window.key.RIGHT:
            pyglet.clock.schedule_interval(self.car.right, 1 / FPS)
        elif symbol == pyglet.window.key.SPACE:
            self.track.create_track(
                self.track.track_vertices[0][0][0], self.track.track_vertices[0][0][1]
            )
            self.track.create_track(
                self.track.track_vertices[0][1][0], self.track.track_vertices[0][1][1]
            )
        elif symbol == pyglet.window.key.S:
            self.track.save(input("Track name: "))
        elif symbol == pyglet.window.key.L:
            self.track.load(input("Track name: "))
        elif symbol == pyglet.window.key.C:
            self.track.clear()
            self.car.reset()
        elif symbol == pyglet.window.key.T:
            self.track.track_visible = not self.track.track_visible
        elif symbol == pyglet.window.key.R:
            self.track.rays_visible = not self.track.rays_visible
        elif symbol == pyglet.window.key.G:
            self.car.god = not self.car.god
        elif symbol == pyglet.window.key.M:
            if self.model_enabled:
                pyglet.clock.unschedule(self.update_from_model)
            else:
                pyglet.clock.schedule_interval(self.update_from_model, 1 / FPS)
            self.model_enabled = not self.model_enabled
        elif symbol == pyglet.window.key.ESCAPE:
            self.close()

    def on_key_release(self, symbol, modifiers):
        if symbol == pyglet.window.key.UP:
            pyglet.clock.unschedule(self.car.forward)
        elif symbol == pyglet.window.key.DOWN:
            pyglet.clock.unschedule(self.car.backward)
        elif symbol == pyglet.window.key.LEFT:
            pyglet.clock.unschedule(self.car.left)
        elif symbol == pyglet.window.key.RIGHT:
            pyglet.clock.unschedule(self.car.right)


##################################################
###################### MAIN ######################
##################################################

if __name__ == "__main__":
    window = Window()
    pyglet.app.run()
