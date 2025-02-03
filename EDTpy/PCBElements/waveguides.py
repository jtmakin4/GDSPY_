from EDTpy import EmptyPath, EmptyGeometry
from EDTpy.validator import *
import gdspy

__all__ = ["CPW"]


class CPW(EmptyPath):
    default_values = {
        'S': 200,
        'W': 150,
        'g': 350, # расстояние между дырками и линией.
        'via_r': Settings.VIA_D/2,
    }

    def _drawing(self, values):
        values['S'] = self.value_quanted(values['S'])
        values['W'] = self.value_quanted(values['W'])

        # заглушка для прямых линий
        if len(self.radiuses) > 0:
            curve_radius = self.radiuses[0]
        else:
            curve_radius = None

        self + gdspy.FlexPath(self.path,
                              values['S'] + 2 * values['W'],
                              corners="circular bend",
                              bend_radius=curve_radius,
                              tolerance=Settings.CURVE_TOLERANCE/10,
                              precision=Settings.CURVE_PRECISION,
                              max_points=Settings.MAX_POLYGON_POINTS,
                              layer=Settings.CPW_LAYER).to_polygonset()

        self - gdspy.FlexPath(self.path,
                              values['S'],
                              corners="circular bend",
                              bend_radius=curve_radius,
                              tolerance=Settings.CURVE_TOLERANCE/10,
                              precision=Settings.CURVE_PRECISION,
                              max_points=Settings.MAX_POLYGON_POINTS,
                              layer=Settings.CPW_LAYER).to_polygonset()

        self.isInverted[Settings.CPW_LAYER] = False

        spots_len_positions = np.arange(0, self.length, 600)
        self.set_spots(length=spots_len_positions, on_curve_segment=True)
        # print(self.spots)
        # отрисовка via holes
        vias = EmptyGeometry()
        via_dist = values['S']/2 + values['W'] + values['g'] + values['via_r']
        vias = vias + gdspy.Round((0, -via_dist),
                                  values['via_r'],
                                  max_points=Settings.MAX_POLYGON_POINTS,
                                  number_of_points=Settings.NUMBER_OF_CIRCLE_POINTS*10,
                                  layer=Settings.VIA_LAYER)
        vias = vias + gdspy.Round((0, via_dist),
                                  values['via_r'],
                                  max_points=Settings.MAX_POLYGON_POINTS,
                                  number_of_points=Settings.NUMBER_OF_CIRCLE_POINTS*10,
                                  layer=Settings.VIA_LAYER)
        vias.add_port((0, 0), 0)

        # размещение via_holes на спотах
        for spot in self.spots:
            vias.merge_with(aim_port=spot, port_num=0)
            self + vias
