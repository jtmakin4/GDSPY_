from EDTpy import EmptyGeometry
from EDTpy.settings import *
import gdspy
import numpy as np

__all__ = ['C_claw_round', 'L_claw_round']


class C_claw_round(EmptyGeometry):
    default_values = {
        'R': 25+46,
        'S': 10,
        'W': 10,
        'gap': 10,
        'angle': 60,
        'port_len': 10,  # must be 0 in real geometry!
        'layer': 0
    }

    def _drawing(self, values):
        self.name = 'C_claw_round'
        self.isInverted[values['layer']] = True

        self + gdspy.Round(
            center=(0, 0),
            radius=values['R'] + values['gap'] + values['S'] + values['W']*2,
            inner_radius=values['R'] + values['gap'],
            initial_angle=(270-values['angle']/2)*np.pi/180,
            final_angle=(270+values['angle']/2)*np.pi/180,
            tolerance=Settings.CURVE_TOLERANCE,
            max_points=Settings.MAX_POLYGON_POINTS,
            number_of_points=Settings.NUMBER_OF_CIRCLE_POINTS,
            layer=values['layer']
        )

        self + gdspy.Rectangle(
            (-values['S'] / 2 - values['W'], -(values['R'] + values['gap'] + values['S'] / 2 + values['W'])),
            (values['S'] / 2 + values['W'], -(values['R'] + values['gap'] + values['S'] + values['W']*2)-values['port_len']),
            layer=values['layer']
        )

        W_angle = values['W']/(values['R'] + values['gap'] + values['S']/2 + values['W'])*180/np.pi

        self.rotate(values['angle'] / 2)
        self - gdspy.Round(
            center=(0, 0),
            radius=values['R'] + values['gap'] + values['S'] + values['W'],
            inner_radius=values['R'] + values['gap'] + values['W'],
            initial_angle=(270 + W_angle) * np.pi / 180,
            final_angle=(270 + values['angle'] - W_angle) * np.pi / 180,
            tolerance=Settings.CURVE_TOLERANCE,
            max_points=Settings.MAX_POLYGON_POINTS,
            number_of_points=Settings.NUMBER_OF_CIRCLE_POINTS,
            layer=values['layer']
        )
        self.rotate(-values['angle']/2)

        self - gdspy.Rectangle(
            (-values['S'] / 2, -(values['R'] + values['gap'] + values['S'] / 2 + values['W'])),
            (values['S'] / 2, -(values['R'] + values['gap'] + values['S'] + values['W'] * 2)-values['port_len']),
            layer=values['layer']
        )

        self.add_port((0, 0), 90)
        self.add_port((0, -(values['R'] + values['gap'] + values['S'] + values['W'] * 2)), 270)


class L_claw_round(EmptyGeometry):
    default_values = {
        'R': 25+46,
        'S': 5,
        'W': 20,
        'gap1': 5,
        'gap2': 10,
        'port_len': 100,
        'layer': 0
    }

    def _drawing(self, values):
        self.name = 'L_claw_round'
        self.isInverted[values['layer']] = True
        max_gap = max(values['gap1'], values['gap2'])

        self + gdspy.Rectangle((-values['S'] / 2 - values['W'], 0),
                               (+values['S'] / 2 + values['W'], values['R'] + max_gap + values['port_len']),
                               layer=values['layer'])

        self - gdspy.Rectangle((-values['S'] / 2, 0),
                               (+values['S'] / 2, values['R'] + max_gap + values['port_len']),
                               layer=values['layer'])

        self - gdspy.Round(
            center=(0, 0),
            radius=values['R'] + values['gap1'],
            initial_angle=0,
            final_angle=90*np.pi/180,
            tolerance=Settings.CURVE_TOLERANCE,
            max_points=Settings.MAX_POLYGON_POINTS,
            number_of_points=Settings.NUMBER_OF_CIRCLE_POINTS,
            layer=values['layer']
        )

        self - gdspy.Round(
            center=(0, 0),
            radius=values['R'] + values['gap2'],
            initial_angle=90 * np.pi / 180,
            final_angle=180 * np.pi / 180,
            tolerance=Settings.CURVE_TOLERANCE,
            max_points=Settings.MAX_POLYGON_POINTS,
            number_of_points=Settings.NUMBER_OF_CIRCLE_POINTS,
            layer=values['layer']
        )

        self.add_port((0, 0), 270)
        self.add_port((0, values['R'] + max_gap + values['port_len']), 90)


if __name__ == '__main__':
    # claw = C_claw_round()
    claw = L_claw_round()
    claw.show()
