from EDTpy import EmptyGeometry
from EDTpy.PCBElements import CPW
from EDTpy.settings import *
import gdspy
import numpy as np

__all__ = ["CoaxCPW", 'ChipCPW']


class CoaxCPW(EmptyGeometry):
    # параметры взяты из interface parameters interface coax-TL.mph
    default_values = {
            "S1": 500,
            "W1": 320,
            "S2": 400,
            "W2": 370,
            "S3": 400,
            "W3": 370,
            "S4": 400,
            "W4": 150,
            "R0": 770,
            "N": 9,
            "r_round": 500,
        }

    def _drawing(self, values):
        self.name = 'Coax-CPW interface'

        via_offset = CPW.default_values['S'] / 2 + CPW.default_values['W'] + \
                     CPW.default_values['g'] + CPW.default_values['via_r']

        init_angle = np.arcsin(via_offset / values['R0'])
        angle_quant = (2 * np.pi - 2 * init_angle) / (values['N'] - 1)

        # length = round((np.cos(init_angle)*values['R0'] + Settings.VIA_DIST)/10, 0)*10
        length = 1100  # отвязываемся от неточности нахождения отверстий, соединяющих землю

        # инвертируем все металлические слои
        for num, layer in Settings.LAYERS.items():
            if layer.is_metal:
                self.isInverted[num] = True

        # отрисовываем все слои
        for key, _ in values.items():
            if 'S' in key:
                S = values[key]
                W = values['W' + key[1:]]
                layer = (int(key[1:]) - 1) * 2

                self + gdspy.Round((0, 0),
                                   radius=S / 2 + W,
                                   max_points=Settings.MAX_POLYGON_POINTS,
                                   number_of_points=Settings.NUMBER_OF_CIRCLE_POINTS,
                                   layer=layer)

                if layer == Settings.CPW_LAYER:
                    self + gdspy.Rectangle((0, -CPW.default_values['S'] / 2 - CPW.default_values['W']),
                                           (length,
                                            CPW.default_values['S'] / 2 + CPW.default_values['W']),
                                           layer=Settings.CPW_LAYER)

                self - gdspy.Round((0, 0),
                                   radius=S / 2,
                                   max_points=Settings.MAX_POLYGON_POINTS,
                                   number_of_points=Settings.NUMBER_OF_CIRCLE_POINTS,
                                   layer=layer)

                if layer == Settings.CPW_LAYER:
                    self - gdspy.Rectangle((0, -CPW.default_values['S'] / 2),
                                           (length, CPW.default_values['S'] / 2),
                                           layer=Settings.CPW_LAYER)

        # рисуем via_holes
        for num, layer in Settings.LAYERS.items():
            if layer.is_metal is None:
                self + gdspy.Round((0, 0),
                                   radius=Settings.VIA_D/2,
                                   max_points=Settings.MAX_POLYGON_POINTS,
                                   number_of_points=Settings.NUMBER_OF_CIRCLE_POINTS,
                                   layer=num)

                for i in range(values['N']):
                    self + gdspy.Round((np.cos(init_angle + angle_quant * i)*values['R0'],
                                        np.sin(init_angle + angle_quant * i)*values['R0']),
                                       radius=Settings.VIA_D/2,
                                       max_points=Settings.MAX_POLYGON_POINTS,
                                       number_of_points=Settings.NUMBER_OF_CIRCLE_POINTS,
                                       layer=num)

        self.add_port((0, 0), 0)
        self.add_port((length, 0), 0)


class ChipCPW(EmptyGeometry):
    # параметры взяты из interface parameters interface coax-TL.mph
    default_values = {
            'H': 14e3,
            "N": 10,
            "TH": 300,  # TH-treshold
            "corner_R": 300/2#1000/2
        }

    def _drawing(self, values):
        values = self.default_values
        self.name = 'Chip-CPW interface'

        # offset = CPW.default_values['S'] + CPW.default_values['W']*2 + \
        #          CPW.default_values['g']*2 + CPW.default_values['via_r']*2
        offset = CPW.default_values['S']*4 + CPW.default_values['W']*2 + \
                 CPW.default_values['g']*2 + CPW.default_values['via_r']*2

        length = 1100  # отвязываемся от неточности нахождения отверстий, соединяющих землю

        # инвертируем все металлические слои и отрисовываем геометрию
        for num, layer in Settings.LAYERS.items():
            if layer.is_metal is True:
                self.isInverted[num] = True
                self + gdspy.Rectangle((-values["H"]/2-values["TH"], -values["H"]/2-values["TH"]),
                                       (values["H"]/2+values["TH"], values["H"]/2+values["TH"]),
                                       layer=num)
                self + gdspy.Round((values["H"]/2, values["H"]/2),
                                   values["corner_R"] + values["TH"],
                                   max_points=Settings.MAX_POLYGON_POINTS,
                                   number_of_points=Settings.NUMBER_OF_CIRCLE_POINTS,
                                   layer=num)

                self + gdspy.Round((values["H"] / 2, -values["H"] / 2),
                                   values["corner_R"] + values["TH"],
                                   max_points=Settings.MAX_POLYGON_POINTS,
                                   number_of_points=Settings.NUMBER_OF_CIRCLE_POINTS,
                                   layer=num)

                self + gdspy.Round((-values["H"] / 2, -values["H"] / 2),
                                   values["corner_R"] + values["TH"],
                                   max_points=Settings.MAX_POLYGON_POINTS,
                                   number_of_points=Settings.NUMBER_OF_CIRCLE_POINTS,
                                   layer=num)

                self + gdspy.Round((-values["H"] / 2, values["H"] / 2),
                                   values["corner_R"] + values["TH"],
                                   max_points=Settings.MAX_POLYGON_POINTS,
                                   number_of_points=Settings.NUMBER_OF_CIRCLE_POINTS,
                                   layer=num)

            elif layer.is_metal is False:
                self + gdspy.Rectangle((-values["H"] / 2, -values["H"] / 2),
                                       (values["H"] / 2, values["H"] / 2),
                                       layer=num)
                self + gdspy.Round((values["H"] / 2, values["H"] / 2),
                                   values["corner_R"],
                                   max_points=Settings.MAX_POLYGON_POINTS,
                                   number_of_points=Settings.NUMBER_OF_CIRCLE_POINTS,
                                   layer=num)
                self + gdspy.Round((values["H"] / 2, -values["H"] / 2),
                                   values["corner_R"],
                                   max_points=Settings.MAX_POLYGON_POINTS,
                                   number_of_points=Settings.NUMBER_OF_CIRCLE_POINTS,
                                   layer=num)

                self + gdspy.Round((-values["H"] / 2, values["H"] / 2),
                                   values["corner_R"],
                                   max_points=Settings.MAX_POLYGON_POINTS,
                                   number_of_points=Settings.NUMBER_OF_CIRCLE_POINTS,
                                   layer=num)

                self + gdspy.Round((-values["H"] / 2, -values["H"] / 2),
                                   values["corner_R"],
                                   max_points=Settings.MAX_POLYGON_POINTS,
                                   number_of_points=Settings.NUMBER_OF_CIRCLE_POINTS,
                                   layer=num)
            else:
                pass

        self.add_port((values["H"]/2+values["TH"]*2+Settings.VIA_D/2, -offset*(values['N']-1)/2), 0)
        for i in range(values['N']-1):
            self.add_port(np.array(self.ports[-1].position)+np.array([0, offset]), 0)

        #self.rotate(45)


if __name__ == '__main__':
    inter = ChipCPW()
    inter.show()
