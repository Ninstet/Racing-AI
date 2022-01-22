##################################################
##################### IMPORTS ####################
##################################################

from dataclasses import asdict
import pyglet
import numpy as np
from time import perf_counter



##################################################
#################### FUNCTIONS ###################
##################################################

def perp(a) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def seg_intersect(a1, a2, b1, b2) :
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1

    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)

    if denom == 0:
        return (np.inf, np.inf)
    else:
        return (num / denom.astype(float))*db + b1



##################################################
##################### CLASSES ####################
##################################################

class Track:

    def __init__(self):
        self.track_vertices = []
        self.tmp_vertex = ()

        self.line_shapes = []
        self.track_shapes = []
        self.temp_shapes = []

        self.visible = True

    def __str__(self):
        return str(self.track_vertices)

    def __repr__(self):
        return str(self.track_vertices)

    def create_track(self, x, y):
        '''
        Adds a vertex to the track.
        '''

        # If the vertex being placed is the first vertex, or the pair of vertices is complete and a new vertex needs to be created
        if self.tmp_vertex == ():
            self.tmp_vertex = (x, y)

        # If the first vertex in the pair has already been placed
        else:
            self.track_vertices.append((self.tmp_vertex, (x, y)))
            self.tmp_vertex = ()
            self.update_shapes()

    def update_shapes(self):
        '''
        Updates the shapes based on the most recent vertex.
        '''

        # If the pair of vertices is complete
        if self.tmp_vertex == ():

            # Draw reward line between two vertices
            new_pair = self.track_vertices[-1]

            line = pyglet.shapes.Line(new_pair[0][0], new_pair[0][1], new_pair[1][0], new_pair[1][1], 3, color=(255, 0, 0))
            line.opacity = 100
            self.line_shapes.append(line)

            # If this is not the first reward line
            if len(self.line_shapes) > 1:

                # Draw joining track triangles
                old_pair = self.track_vertices[-2]

                shape_1 = pyglet.shapes.Triangle(new_pair[0][0], new_pair[0][1], new_pair[1][0], new_pair[1][1], old_pair[0][0], old_pair[0][1], color=(145, 145, 145))
                shape_2 = pyglet.shapes.Triangle(new_pair[1][0], new_pair[1][1], old_pair[0][0], old_pair[0][1], old_pair[1][0], old_pair[1][1], color=(145, 145, 145))
                shape_1.opacity = 100
                shape_2.opacity = 120

                self.track_shapes.append(shape_1)
                self.track_shapes.append(shape_2)

                # Draw track edges
                self.line_shapes.append(pyglet.shapes.Line(new_pair[0][0], new_pair[0][1], old_pair[0][0], old_pair[0][1], 3, color=(0, 0, 0)))
                self.line_shapes.append(pyglet.shapes.Line(new_pair[1][0], new_pair[1][1], old_pair[1][0], old_pair[1][1], 3, color=(0, 0, 0)))

    def distance_to_nearest_intersection(self, a1, a2):
        '''
        Finds the distance to the nearest intersection point of an arbitrary line with any line on the track.
        '''

        # If there is at least 1 pair of track vertices
        if len(self.track_vertices) > 0:
            track_vertices = np.array(self.track_vertices)

            # 2 lines below take 0.8ms (was 5.4ms previously)
            intersections_1, distances_1 = self.calculate_intersections(a1, a2, track_vertices[:, 0])
            intersections_2, distances_2 = self.calculate_intersections(a1, a2, track_vertices[:, 1])

            intersections = intersections_1 + intersections_2
            distances = distances_1 + distances_2

            # If at least 1 intersection has been found
            if len(distances) > 0:
                sorted_intersections = []
                sorted_distances = []

                for map in sorted(zip(distances, intersections)):
                    sorted_intersections.append(map[1])
                    sorted_distances.append(map[0])

                P1 = sorted_intersections[0]
                Ps = [P1]

                for P in sorted_intersections[1:]:
                    if (P1[0] > a1[0] and P[0] < a1[0]) or (P1[1] > a1[1] and P[1] < a1[1]) or (P1[0] < a1[0] and P[0] > a1[0]) or (P1[1] < a1[1] and P[1] > a1[1]):
                        Ps.append(P)
                        break

                for P in Ps:
                    # 2 lines below take 0.3ms
                    self.temp_shapes.append(pyglet.shapes.Line(a1[0], a1[1], P[0], P[1], 2, color=(0, 255, 0)))
                    self.temp_shapes.append(pyglet.shapes.Circle(P[0], P[1], 5, color=(0, 145, 0)))

                return sorted_distances[0]

            else:
                return None

    def calculate_intersections(self, a1, a2, lines):
        '''
        Calculates all intersection points between an arbitrary line with any line on the track.
        '''

        intersections = []
        distances = []

        for i in range(len(lines) - 1):
            b1 = lines[i]
            b2 = lines[i + 1]

            P = seg_intersect(a1, a2, b1, b2)

            if P[0] != np.inf:
                if P[0] > min(b1[0], b2[0]) and P[0] < max(b1[0], b2[0]) and P[1] > min(b1[1], b2[1]) and P[1] < max(b1[1], b2[1]):
                    intersections.append(P)
                    distances.append(np.sqrt(np.sum((P - a1)**2)))

        return intersections, distances

    def clear(self):
        '''
        Clear / reset the track.
        '''

        for line in self.line_shapes:
            line.delete()
        for shape in self.track_shapes:
            shape.delete()
        for shape in self.temp_shapes:
            shape.delete()

        self.track_vertices = []
        self.tmp_vertex = ()
        self.line_shapes = []
        self.track_shapes = []
        self.temp_shapes = []

    def save(self, filename):
        '''
        Save the track to a text file.
        '''

        if self.tmp_vertex == ():

            # Convert 3D array into 2D array
            track_vertices = np.array(self.track_vertices)
            track_vertices_raw = track_vertices.reshape(track_vertices.shape[0], -1)

            # Save file
            np.savetxt(filename + ".txt", track_vertices_raw, delimiter=', ', fmt='%s')

        else:
            print("Track incomplete!")
    
    def load(self, filename):
        '''
        Load the track from a text file.
        '''

        self.clear()

        # Load file
        track_vertices_raw = np.loadtxt(filename + ".txt", delimiter=', ', dtype=float).astype(int)

        # Convert 2D array into 3D array
        track_vertices = track_vertices_raw.reshape(track_vertices_raw.shape[0], 2, 2)

        for pair in track_vertices:
            self.create_track(pair[0][0], pair[0][1])
            self.create_track(pair[1][0], pair[1][1])