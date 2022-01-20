import pyglet
import numpy as np

class Track:
    def __init__(self, vertices = []):
        self.track_vertices = []
        self.lines = []
        self.track = []

        for vertex in vertices:
            self.create_track(vertex[0], vertex[1])

    def create_track(self, x, y):
        self.track_vertices.append((x, y))

        if len(self.track_vertices) % 2 == 0 and len(self.track_vertices) > 0:
            line = pyglet.shapes.Line(self.track_vertices[-1][0], self.track_vertices[-1][1], self.track_vertices[-2][0], self.track_vertices[-2][1], 3, color=(255, 0, 0))
            line.opacity = 100
            self.lines.append(line)

            if len(self.lines) > 1:
                shape_1 = pyglet.shapes.Triangle(self.track_vertices[-1][0], self.track_vertices[-1][1], self.track_vertices[-2][0], self.track_vertices[-2][1], self.track_vertices[-3][0], self.track_vertices[-3][1], color=(145, 145, 145))
                shape_2 = pyglet.shapes.Triangle(self.track_vertices[-2][0], self.track_vertices[-2][1], self.track_vertices[-3][0], self.track_vertices[-3][1], self.track_vertices[-4][0], self.track_vertices[-4][1], color=(145, 145, 145))
                shape_1.opacity = 100
                shape_2.opacity = 120

                self.track.append(shape_1)
                self.track.append(shape_2)

                self.lines.append(pyglet.shapes.Line(self.track_vertices[-1][0], self.track_vertices[-1][1], self.track_vertices[-3][0], self.track_vertices[-3][1], 3, color=(0, 0, 0)))
                self.lines.append(pyglet.shapes.Line(self.track_vertices[-2][0], self.track_vertices[-2][1], self.track_vertices[-4][0], self.track_vertices[-4][1], 3, color=(0, 0, 0)))

        print(self.track_vertices)
    
    def save(self, filename):
        np.savetxt(filename + ".txt", np.array(self.track_vertices), fmt='%s', delimiter=', ')
    
    def clear(self):
        for line in self.lines:
            line.delete()
        for shape in self.track:
            shape.delete()

        self.track_vertices = []
        self.lines = []
        self.track = []
    
    def load(self, filename):
        self.clear()

        vertices = np.loadtxt(filename + ".txt", delimiter=', ')

        for vertex in vertices:
            self.create_track(vertex[0], vertex[1])