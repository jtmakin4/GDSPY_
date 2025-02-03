from EDTpy import EmptySketch
from EDTpy.settings import *
import gdspy
import numpy as np

__all__ = ["RectangularSketch", "CircularSketch"]


class RectangularSketch(EmptySketch):
    default_values = {
        'width': 55e3,
        'height': 130e3,
    }

    def _drawing(self, values):
        self.name = 'Rectangle Sketch'
        values['width'] = self.value_quanted(values['width'])
        values['height'] = self.value_quanted(values['height'])
        for layer in range(6, 7):
            self + gdspy.Rectangle((-values['width']/2, -values['height']/2),
                                   (+values['width']/2, +values['height']/2),
                                   layer=layer)


class CircularSketch(EmptySketch):
    default_values = {
        'diameter': 55e3}

    def _drawing(self, values):
        self.name = 'Circular Sketch'
        values['diameter'] = self.value_quanted(values['diameter'])
        for layer in range(0, 7):
            self + gdspy.Round((0, 0), values['diameter']/2,
                               max_points=2**20,
                               number_of_points=100,
                               initial_angle=0,
                               final_angle=np.pi/2,
                               layer=layer)