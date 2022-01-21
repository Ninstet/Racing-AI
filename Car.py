import pyglet
import numpy as np

ACCELERATION = 0.8
ROTATION_SPEED = 30
FPS = 60

class Car:

    def __init__(self, x, y, friction, vector, track):
        '''
        Initialises a car object which has physics and can detect collisions
        '''

        self.start_pos = (x, y)
        self.friction = friction
        self.vector = vector
        self.track = track

        self.displacement = np.array([1.0, 0.0])
        self.velocity = np.array([0.0, 0.0])

        self.speed = 0
        self.angular_speed = 0
        self.bearing = 0

        green_car_image = pyglet.resource.image('green_car.png')
        green_car_image.anchor_x = green_car_image.width // 2
        green_car_image.anchor_y = green_car_image.height // 2

        self.sprite = pyglet.sprite.Sprite(green_car_image, x=x, y=y)
        self.sprite.scale = 0.05

        pyglet.clock.schedule_interval(self.physics, 1 / FPS)

    def physics(self, dt):
        '''
        Physics engine for the car.
        '''

        # Acceleration
        self.velocity += self.displacement * self.speed

        self.velocity *= self.friction # Lateral friction
        self.speed *= self.friction # Drag

        self.sprite.x += self.velocity[0] # Update x position
        self.sprite.y += self.velocity[1] # Update y position

        # Steering
        self.bearing += self.angular_speed * abs(self.speed)**0.4

        self.angular_speed *= self.friction
        self.displacement = np.array([np.cos(np.radians(self.bearing)), -np.sin(np.radians(self.bearing))]) # Normalise displacement vector

        self.sprite.rotation = self.bearing # Update sprite orientation

        # Graphics
        self.vector.position = (self.sprite.x, self.sprite.y, self.sprite.x + self.velocity[0] * 10, self.sprite.y + self.velocity[1] * 10)

        # Collisions
        pos = np.array((self.sprite.x, self.sprite.y))
        P, distance = self.track.distance_to_intersection(pos, pos + self.displacement)

        print(pos + self.displacement)

        if distance != None:
            if distance < 10:
                self.displacement = np.array([1.0, 0.0])
                self.velocity = np.array([0.0, 0.0])

                self.speed = 0
                self.angular_speed = 0
                self.bearing = 0

                self.sprite.position = self.start_pos
                self.sprite.rotation = self.bearing

    # def has_collided(self):
        



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

    def right(self, dt):
        self.angular_speed += ROTATION_SPEED * dt

    def left(self, dt):
        self.angular_speed -= ROTATION_SPEED * dt



    def draw(self):
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.sprite.draw()