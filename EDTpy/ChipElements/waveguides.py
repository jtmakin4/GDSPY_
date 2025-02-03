import EDTpy
from EDTpy import EmptyGeometry
from EDTpy import EmptyPath
from EDTpy.settings import *
import gdspy
import numpy as np

__all__ = ['CPW']


class CPW(EmptyPath):
    default_values = {
        'S': 20,
        'W': 12,
        'layer': 0,
    }

    def _drawing(self, values):
        self.name = 'CPW'
        # self.isInverted[values['layer']] = True

        # заглушка для прямых линий
        if len(self.radiuses) > 0:
            curve_radius = self.radiuses[0]
        else:
            curve_radius = None

        self + gdspy.FlexPath(self.path,
                              values['S'] + 2 * values['W'],
                              corners="circular bend",
                              bend_radius=curve_radius,
                              tolerance=Settings.CURVE_TOLERANCE,
                              precision=Settings.CURVE_PRECISION,
                              max_points=Settings.MAX_POLYGON_POINTS,
                              layer=values['layer']).to_polygonset()

        self - gdspy.FlexPath(self.path,
                              values['S'],
                              corners="circular bend",
                              bend_radius=curve_radius,
                              tolerance=Settings.CURVE_TOLERANCE,
                              precision=Settings.CURVE_PRECISION,
                              max_points=Settings.MAX_POLYGON_POINTS,
                              layer=values['layer']).to_polygonset()
        self.add_port(self.path[0],
                      round(np.angle(
                          (np.array(self.path[0]) - np.array(self.path[1]))[0]
                          + 1j*(np.array(self.path[0]) - np.array(self.path[1]))[1]
                      ) * 180 / np.pi)
                      )
        self.add_port(self.path[-1],
                      round(np.angle(
                          (np.array(self.path[-1]) - np.array(self.path[-2]))[0]
                          + 1j * (np.array(self.path[-1]) - np.array(self.path[-2]))[1]
                      ) * 180 / np.pi)
                      )


if __name__ == '__main__':

    res_r =1900
    res_l = 1200

    path = [
        (1000, 0),
        (2000, 0)
    ]
    path1 = [
        (res_l, -2070),
        (res_l, -1240),
        (res_r, -1240),
        (res_r, -940),
        (res_l, -940),
        (res_l, -640),
        (res_r, -640),
        (res_r, -340),
        (res_l, -340),
        (res_l, -40),
        (1800, -40)
    ]
    # feedline = CPW(path, 150, S=10, W=12) # сама линия
    resonator = CPW(path1, 150, S=10, W=6) # resonator
    resonator.show()
    # resonator + gdspy.Rectangle((1800, -29), (1810, -51))
    #
    # sketch = EDTpy.EmptySketch()
    # sketch.add_geometry(feedline)
    # sketch.add_geometry(resonator)
    # # cpw.rotate(45)
    # # cpw.translate((-1000, -1000))
    # print(resonator.length)
    # sketch.assemble()
    # sketch.show()
    # sketch.save_gds('resonator')



