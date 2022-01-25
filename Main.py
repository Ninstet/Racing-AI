##################################################
##################### IMPORTS ####################
##################################################

import pyglet
import numpy as np
# pyglet.options['audio'] = ('openal', 'pulse', 'directsound', 'silent')
# import pyglet.media

from Car import Car
from Track import Track

FPS = 60



##################################################
##################### CLASSES ####################
##################################################

class Window(pyglet.window.Window):

    def __init__(self):
        super(Window, self).__init__(width=1280, height=720)
        self.set_visible()

        self.track = Track()
        self.track.load("track_1")

        self.drag = pyglet.shapes.Line(0, 0, 0, 0, 3, color=(250, 30, 30))
        self.drag.opacity = 250

        self.car = Car(400, 200, 0.95, self.track)

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
        
        pyglet.text.Label(f"Score: {str(self.car.target_reward_gate)}", color=(0, 0, 0, 255), font_name='Arial', font_size=16, x=1150, y=50, anchor_x='center', anchor_y='center').draw()

    def on_mouse_press(self, x, y, button, modifiers):
        self.track.create_track(x, y)
        self.drag.position = (x, y, x, y)

    def on_mouse_release(self, x, y, button, modifiers):
        self.track.create_track(x, y)
        self.drag.position = (0, 0, 0, 0)

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
            self.track.create_track(self.track.track_vertices[0][0][0], self.track.track_vertices[0][0][1])
            self.track.create_track(self.track.track_vertices[0][1][0], self.track.track_vertices[0][1][1])
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

if __name__ == '__main__':
    window = Window()
    pyglet.app.run()