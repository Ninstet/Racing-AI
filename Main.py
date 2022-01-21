import pyglet
import numpy as np

from Car import Car
from Track import Track

FPS = 60

window = pyglet.window.Window(width=1280, height=720)
label = pyglet.text.Label('Hello, World!', font_name='Times New Roman', font_size=36, x=window.width//2, y=window.height//2, anchor_x='center', anchor_y='center')

track = Track()
track.load("track_1")

drag = pyglet.shapes.Line(0, 0, 0, 0, 3, color=(250, 30, 30))
drag.opacity = 250

# EVENTS

@window.event
def on_mouse_press(x, y, button, modifiers):
    track.create_track(x, y)
    drag.position = (x, y, x, y)

@window.event
def on_mouse_release(x, y, button, modifiers):
    track.create_track(x, y)
    drag.position = (0, 0, 0, 0)

@window.event
def on_mouse_drag(x, y, dx, dy, button, modifiers):
    drag.x2 = x
    drag.y2 = y

@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.UP:
        pyglet.clock.schedule_interval(car.forward, 1 / FPS)
    elif symbol == pyglet.window.key.DOWN:
        pyglet.clock.schedule_interval(car.backward, 1 / FPS)
    elif symbol == pyglet.window.key.LEFT:
        pyglet.clock.schedule_interval(car.left, 1 / FPS)
    elif symbol == pyglet.window.key.RIGHT:
        pyglet.clock.schedule_interval(car.right, 1 / FPS)
    elif symbol == pyglet.window.key.SPACE:
        track.create_track(track.track_vertices[0][0][0], track.track_vertices[0][0][1])
        track.create_track(track.track_vertices[0][1][0], track.track_vertices[0][1][1])
    elif symbol == pyglet.window.key.S:
        track.save(input("Track name: "))
    elif symbol == pyglet.window.key.L:
        track.load(input("Track name: "))
    elif symbol == pyglet.window.key.C:
        track.clear()

@window.event
def on_key_release(symbol, modifiers):
    if symbol == pyglet.window.key.UP:
        pyglet.clock.unschedule(car.forward)
    elif symbol == pyglet.window.key.DOWN:
        pyglet.clock.unschedule(car.backward)
    elif symbol == pyglet.window.key.LEFT:
        pyglet.clock.unschedule(car.left)
    elif symbol == pyglet.window.key.RIGHT:
        pyglet.clock.unschedule(car.right)





# @window.event
# def on_mouse_release(x, y, button, modifiers):

vector = pyglet.shapes.Line(100, 100, 200, 200, 3, color=(250, 30, 30))
vector.opacity = 250

@window.event
def on_draw():
    window.clear()
    label.draw()

    for i in track.track_shapes:
        i.draw()
    for i in track.line_shapes:
        i.draw()
    for i in track.temp_shapes:
        i.draw()

    car.draw()
    vector.draw()
    drag.draw()

car = Car(400, 200, 0.95, vector, track)
pyglet.app.run()