from EDTpy import EmptyGeometry
from EDTpy.settings import *
import gdspy
import numpy as np

__all__ = ['CoaxialCapacitor']


class CoaxialCapacitor(EmptyGeometry):
    default_values = {
        "S": 50,
        "W": 46,
        "layer": 0,
        "z_lines_angle": 60,
    }

    def _drawing(self, values):
        self.name = "CoaxialCapacitor"
        s = values['S']
        w = values['W']
        layer = values['layer']

        self.isInverted[layer] = True

        self + gdspy.Round((0, 0),
                           radius=s/2+w,
                           max_points=Settings.MAX_POLYGON_POINTS,
                           number_of_points=Settings.NUMBER_OF_CIRCLE_POINTS,
                           layer=layer)

        self - gdspy.Round((0, 0),
                           radius=s / 2,
                           max_points=Settings.MAX_POLYGON_POINTS,
                           number_of_points=Settings.NUMBER_OF_CIRCLE_POINTS,
                           layer=layer)

        # drawing claw and Z_lines ports
        for i in range(0, 4):
            self.add_port((0, 0), 90*i)

        self.add_port((0, 0), 90 - values['z_lines_angle'] / 2)
        self.add_port((0, 0), 90 + values['z_lines_angle'] / 2)

        # drawing JJ ports
        n_inner = 2
        for i in range(0, n_inner):
            self.add_port((0, s/2), 90)
            self.ports[-1].rotate(360/n_inner*i, center=self.position)


if __name__ == '__main__':
    cap = CoaxialCapacitor()
    cap.rotate(45)
    cap.translate((1000, 0))
    cap.show()


