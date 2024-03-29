##################################################
##################### IMPORTS ####################
##################################################


import numpy as np
import pyglet


##################################################
#################### FUNCTIONS ###################
##################################################


def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def seg_intersect(a1, a2, b1, b2):
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1

    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)

    if denom == 0:
        return (np.inf, np.inf)
    else:
        return (num / denom.astype(float)) * db + b1


##################################################
##################### CLASSES ####################
##################################################


class Track:
    def __init__(self, track=None):
        self.track_vertices = []
        self.tmp_vertex = ()

        self.wall_shapes = []
        self.wall_batch = pyglet.graphics.Batch()
        self.gate_shapes = []
        self.gate_batch = pyglet.graphics.Batch()
        self.road_shapes = []
        self.road_batch = pyglet.graphics.Batch()
        self.temp_shapes = []
        self.temp_batch = pyglet.graphics.Batch()

        self.track_visible = True
        self.rays_visible = True

        if track != None:
            self.load(track)

    def __str__(self):
        return str(self.track_vertices)

    def __repr__(self):
        return str(self.track_vertices)

    def create_track(self, x, y):
        """
        Adds a vertex to the track.
        """

        # If the vertex being placed is the first vertex, or the pair of vertices is complete and a new vertex needs to be created
        if self.tmp_vertex == ():
            self.tmp_vertex = (x, y)

        # If the first vertex in the pair has already been placed
        else:
            self.track_vertices.append((self.tmp_vertex, (x, y)))
            self.tmp_vertex = ()
            self.update_shapes()

    def update_shapes(self):
        """
        Updates the shapes based on the most recent vertex.
        """

        # If the pair of vertices is complete
        if self.tmp_vertex == ():
            # Draw reward line between two vertices
            new_pair = self.track_vertices[-1]

            line = pyglet.shapes.Line(
                new_pair[0][0],
                new_pair[0][1],
                new_pair[1][0],
                new_pair[1][1],
                3,
                color=(255, 0, 0),
                batch=self.gate_batch,
            )
            line.opacity = 100
            self.gate_shapes.append(line)

            # If this is not the first reward line
            if len(self.gate_shapes) > 1:
                # If new pair is original pair, delete the duplicate gate
                if new_pair == self.track_vertices[0]:
                    self.gate_shapes.pop()

                # Draw joining track triangles
                old_pair = self.track_vertices[-2]

                shape_1 = pyglet.shapes.Triangle(
                    new_pair[0][0],
                    new_pair[0][1],
                    new_pair[1][0],
                    new_pair[1][1],
                    old_pair[0][0],
                    old_pair[0][1],
                    color=(145, 145, 145),
                    batch=self.road_batch,
                )
                shape_2 = pyglet.shapes.Triangle(
                    new_pair[1][0],
                    new_pair[1][1],
                    old_pair[0][0],
                    old_pair[0][1],
                    old_pair[1][0],
                    old_pair[1][1],
                    color=(145, 145, 145),
                    batch=self.road_batch,
                )
                shape_1.opacity = 100
                shape_2.opacity = 120

                self.road_shapes.append(shape_1)
                self.road_shapes.append(shape_2)

                # Draw track edges
                self.wall_shapes.append(
                    pyglet.shapes.Line(
                        new_pair[0][0],
                        new_pair[0][1],
                        old_pair[0][0],
                        old_pair[0][1],
                        3,
                        color=(0, 0, 0),
                        batch=self.wall_batch,
                    )
                )
                self.wall_shapes.append(
                    pyglet.shapes.Line(
                        new_pair[1][0],
                        new_pair[1][1],
                        old_pair[1][0],
                        old_pair[1][1],
                        3,
                        color=(0, 0, 0),
                        batch=self.wall_batch,
                    )
                )

    def distance_to_reward_gate(self, a1, a2, gate):
        """
        Returns the distance to a given reward gate along a particular vector.
        """

        # Check there is at least 1 pair of track vertices
        if len(self.track_vertices) < 2:
            return None

        track_vertices = np.array(self.track_vertices)

        # Calculate intersections and distances to the reward gates
        intersections, distances = self.compute_intersections(
            a1, a2, track_vertices[(gate + 1) % len(track_vertices), :]
        )

        # Update shape colours to indicate next reward gate
        self.gate_shapes[gate % len(self.gate_shapes)].color = (255, 0, 0)
        self.gate_shapes[(gate + 1) %
                         len(self.gate_shapes)].color = (0, 0, 255)

        # Check there is at least 1 reward gate
        if len(distances) == 0:
            return None

        self.temp_shapes.append(
            pyglet.shapes.Line(
                a1[0],
                a1[1],
                intersections[0][0],
                intersections[0][1],
                2,
                color=(0, 0, 220),
                batch=self.temp_batch,
            )
        )
        self.temp_shapes.append(
            pyglet.shapes.Circle(
                intersections[0][0],
                intersections[0][1],
                5,
                color=(0, 0, 145),
                batch=self.temp_batch,
            )
        )

        return distances[0]

    def get_intersections(self, a1, a2):
        """
        Finds the positions and distances to the nearest intersection points along a line.
        """

        # Check there is at least 1 pair of track vertices
        if len(self.track_vertices) < 1:
            return [], []

        track_vertices = np.array(self.track_vertices)

        # Calculate intersections and distances for inside and outside barriers of track
        intersections_1, distances_1 = self.compute_intersections(
            a1, a2, track_vertices[:, 0]
        )
        intersections_2, distances_2 = self.compute_intersections(
            a1, a2, track_vertices[:, 1]
        )

        # Combine intersections and distances
        intersections = intersections_1 + intersections_2
        distances = distances_1 + distances_2

        # Check at least 1 intersection has been found
        if len(distances) < 1:
            return [], []

        sorted_intersections = []
        sorted_distances = []

        # Sort intersections and distances from smallest distance first
        for map in sorted(zip(distances, intersections)):
            sorted_intersections.append(map[1])
            sorted_distances.append(map[0])

        closest_intersections = [sorted_intersections[0]]
        closest_distances = [sorted_distances[0]]

        # Find intersection point that is opposite to the closest point
        for i in range(len(sorted_intersections) - 1):
            P = sorted_intersections[i + 1]

            if (
                (closest_intersections[0][0] > a1[0] and P[0] < a1[0])
                or (closest_intersections[0][1] > a1[1] and P[1] < a1[1])
                or (closest_intersections[0][0] < a1[0] and P[0] > a1[0])
                or (closest_intersections[0][1] < a1[1] and P[1] > a1[1])
            ):
                closest_intersections.append(P)
                closest_distances.append(sorted_distances[i + 1])

                break

        # Draw intersection detection shapes
        for P in closest_intersections:
            self.temp_shapes.append(
                pyglet.shapes.Line(
                    a1[0],
                    a1[1],
                    P[0],
                    P[1],
                    2,
                    color=(0, 220, 0),
                    batch=self.temp_batch,
                )
            )
            self.temp_shapes.append(
                pyglet.shapes.Circle(
                    P[0], P[1], 5, color=(0, 145, 0), batch=self.temp_batch
                )
            )

        # Return closest intersection distance
        return closest_intersections, closest_distances

    def compute_intersections(self, a1, a2, lines):
        """
        Calculates all intersection points between an arbitrary line with any line on the track.
        """

        if len(lines) < 1:
            return [], []

        intersections = []
        distances = []

        # For all lines to compute intersections on
        for i in range(len(lines) - 1):
            b1 = lines[i]
            b2 = lines[i + 1]

            # Calculate intersection based on 4 points on 2 lines
            P = seg_intersect(a1, a2, b1, b2)

            # Find intersection points that lie on the line (not off the line)
            if P[0] != np.inf:
                if (
                    P[0] > min(b1[0], b2[0])
                    and P[0] < max(b1[0], b2[0])
                    and P[1] > min(b1[1], b2[1])
                    and P[1] < max(b1[1], b2[1])
                ):
                    intersections.append(P)
                    distances.append(np.sqrt(np.sum((P - a1) ** 2)))

        return intersections, distances

    def clear(self):
        """
        Clear / reset the track.
        """

        for shape in self.wall_shapes:
            shape.delete()
        for shape in self.gate_shapes:
            shape.delete()
        for shape in self.road_shapes:
            shape.delete()
        for shape in self.temp_shapes:
            shape.delete()

        self.track_vertices = []
        self.tmp_vertex = ()

        self.wall_shapes = []
        self.wall_batch = pyglet.graphics.Batch()
        self.gate_shapes = []
        self.gate_batch = pyglet.graphics.Batch()
        self.road_shapes = []
        self.road_batch = pyglet.graphics.Batch()
        self.temp_shapes = []
        self.temp_batch = pyglet.graphics.Batch()

    def save(self, filename):
        """
        Save the track to a text file.
        """

        if self.tmp_vertex == ():
            # Convert 3D array into 2D array
            track_vertices = np.array(self.track_vertices)
            track_vertices_raw = track_vertices.reshape(
                track_vertices.shape[0], -1)

            # Save file
            np.savetxt(filename, track_vertices_raw, delimiter=", ", fmt="%s")

        else:
            print("Track incomplete!")

    def load(self, filename):
        """
        Load the track from a text file.
        """

        self.clear()

        # Load file
        track_vertices_raw = np.genfromtxt(
            filename, delimiter=", ", dtype=float
        ).astype(int)

        # Convert 2D array into 3D array
        track_vertices = track_vertices_raw.reshape(
            track_vertices_raw.shape[0], 2, 2)

        for pair in track_vertices:
            self.create_track(pair[0][0], pair[0][1])
            self.create_track(pair[1][0], pair[1][1])
