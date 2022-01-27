##################################################
##################### IMPORTS ####################
##################################################

import pyglet
import numpy as np
from time import perf_counter

ACCELERATION = 0.8
ROTATION_SPEED = 30
FPS = 60



##################################################
##################### CLASSES ####################
##################################################

class Car:

    def __init__(self, x, y, friction, track):
        '''
        Initialises a car object which has physics and can detect collisions
        '''

        self.start_pos = (x, y)
        self.friction = friction
        self.track = track

        self.god = False

        green_car_image = pyglet.resource.image('green_car.png')
        green_car_image.anchor_x = green_car_image.width // 2
        green_car_image.anchor_y = green_car_image.height // 2

        self.sprite = pyglet.sprite.Sprite(green_car_image, x=x, y=y)
        self.sprite.scale = 0.05

        self.vector = pyglet.shapes.Line(100, 100, 200, 200, 3, color=(250, 30, 30))
        self.vector.opacity = 250

        self.reset()
        self.check_sensors()

        pyglet.clock.schedule_interval(self.physics, 1 / FPS)

    def physics(self, dt):
        '''
        Physics engine for the car.
        '''

        # Acceleration
        self.velocity += self.displacement * self.speed

        self.velocity *= self.friction # Lateral friction
        self.speed *= self.friction # Drag

        self.pos += self.velocity

        # Steering
        self.bearing += self.angular_speed * abs(self.speed)**0.4

        self.displacement = np.array([np.cos(np.radians(self.bearing)), -np.sin(np.radians(self.bearing))]) # Normalise displacement vector
        self.angular_speed *= self.friction

        # Graphics
        self.sprite.position = self.pos
        self.sprite.rotation = self.bearing
        self.vector.position = (self.pos[0], self.pos[1], self.pos[0] + self.velocity[0] * 10, self.pos[1] + self.velocity[1] * 10)

        # Collisions
        for shape in self.track.temp_shapes:
            shape.delete()

        self.track.temp_shapes = []

        collision = self.check_sensors()
        distance = self.track.distance_to_reward_gate(self.pos, self.pos + self.displacement, self.target_reward_gate)

        if distance != None:
            if distance < 15:
                self.target_reward_gate += 1

        return collision

    def check_sensors(self):
        '''
        Checks if the car has collided with any of the lines.
        '''
        for i in np.arange(0, 180, 30):
            vector = np.array([np.cos(np.radians(i + self.bearing)), -np.sin(np.radians(i + self.bearing))])

            intersections, distances = self.track.get_intersections(self.pos, self.pos + vector)

            if len(distances) > 0:
                if distances[0] < 15:
                    if self.god == False:
                        for shape in self.track.gate_shapes:
                            shape.color = (255, 0, 0)
                        self.reset()

                        return True

            self.sensors[2 * (i // 30)] = distances[0]
            self.sensors[(2 * (i // 30)) + 1] = distances[1]

        return False

    def reset(self):
        '''
        Reset the cars position, velocity and displacement.
        '''

        self.pos = self.start_pos
        
        self.displacement = np.array([1.0, 0.0])
        self.velocity = np.array([0.0, 0.0])

        self.speed = 0
        self.angular_speed = 0
        self.bearing = 0
        
        self.sensors = np.zeros(2 * (180 // 30))
        self.target_reward_gate = 0

        self.sprite.position = self.pos
        self.sprite.rotation = self.bearing



    def forward(self, dt):
        if self.speed < 0:
            self.speed += ACCELERATION * dt * 0.1
        else:
            self.speed += ACCELERATION * dt

    def backward(self, dt):
        if self.speed < 0:
            self.speed -= ACCELERATION * dt * 0.1
        else:
            self.speed -= ACCELERATION * dt

    def left(self, dt):
        self.angular_speed -= ROTATION_SPEED * dt

    def right(self, dt):
        self.angular_speed += ROTATION_SPEED * dt



    def draw(self):
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.sprite.draw()