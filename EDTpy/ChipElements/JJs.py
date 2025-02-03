from EDTpy import EmptyGeometry
from EDTpy.settings import *
import gdspy
import numpy as np

__all__ = ['JJ_Ileg', 'JJ_Lleg', 'JJ_Tfork', 'JJ_Yfork', 'DCsquid', 'JJ_double']


class JJ_Ileg(EmptyGeometry):
    default_values = {
        'base_width': 5,
        'length': 20,

        'patch_offset': 3,

        'JJ_width': 0.5,
        "layer": 1,
    }

    def _drawing(self, values):
        self.name = 'JJ_Ileg'

        self + gdspy.Rectangle(
            (-values['JJ_width'] / 2, 0),
            (values['JJ_width'] / 2, values['length']),
            layer=values['layer']
        )

        self + gdspy.Rectangle(
            (-values['base_width'] / 2, 0),
            (values['base_width'] / 2, values['length'] - values['patch_offset'] - (values['base_width']-values['JJ_width'])/2),
            layer=values['layer']
        )

        patch = gdspy.Polygon(
            [
                (values['JJ_width'] / 2,
                 values['length'] - values['patch_offset'] - (values['base_width']-values['JJ_width'])/2),

                (values['JJ_width'] / 2,
                 values['length'] - values['patch_offset']),

                (values['base_width'] / 2,
                 values['length'] - values['patch_offset'] - (values['base_width'] - values['JJ_width']) / 2),

            ],
            layer=values['layer']
        )

        self + patch
        self + patch.mirror(self.position, self.position + np.array([0, 1]))
        #
        self.add_port((0, 0), 270)

        self.add_port((0,
                       values['length']), 90)


class JJ_Lleg(EmptyGeometry):
    default_values = {
        'base_width': 5,
        'base_length': 20,

        'leg_length': 10,
        'patch_offset': 3,

        'JJ_width': 0.5,
        'bridge': 1,
        "layer": 1,
    }

    def _drawing(self, values):
        self.name = 'JJ_Lleg'

        self + gdspy.Rectangle(
            (-values['base_width']/2, 0),
            (values['base_width']/2, values['base_length']),
            layer=values['layer']
        )

        self + gdspy.Rectangle(
            (-values['leg_length'], values['base_length'] - values['JJ_width']),
            (0, values['base_length']),
            layer=values['layer']
        )

        patch = gdspy.Polygon(
            [
                (-values['base_width']/2,
                 values['base_length'] - values['JJ_width']),
                (-values['leg_length'] + values['patch_offset'],
                 values['base_length'] - values['JJ_width']),
                (-values['base_width'] / 2,
                 values['base_length'] - values['JJ_width'] - (values['leg_length'] - values['patch_offset'] - values['base_width']/2)),

            ],
            layer=values['layer']
        )

        self + patch

        self.add_port((0, 0), 270)
        self.add_port((-values['leg_length'] + values['bridge'],
                       values['base_length'] + values['bridge']), 90)


class JJ_Tfork(EmptyGeometry):
    default_values = {
        'base_width': 5,
        'base_length': 20,

        'fork_width': 20,
        'patch_offset': 3,

        'JJ_width': 0.5,
        'bridge': 1,
        "layer": 1,
    }

    def _drawing(self, values):
        self.name = 'JJ_Tfork'

        self + gdspy.Rectangle(
            (-values['base_width']/2, 0),
            (values['base_width']/2, values['base_length']),
            layer=values['layer']
        )

        self + gdspy.Rectangle(
            (-values['fork_width'] / 2, values['base_length'] - values['JJ_width']),
            (values['fork_width'] / 2, values['base_length']),
            layer=values['layer']
        )

        patch = gdspy.Polygon(
            [
                (-values['base_width']/2,
                 values['base_length'] - values['JJ_width']),
                (-values['fork_width']/2 + values['patch_offset'],
                 values['base_length'] - values['JJ_width']),
                (-values['base_width'] / 2,
                 values['base_length'] - values['JJ_width'] - (values['fork_width']/2 - values['patch_offset'] - values['base_width']/2)),

            ],
            layer=values['layer']
        )

        self + patch
        self + patch.mirror(self.position, self.position + np.array([0, 1]))

        self.add_port((0, 0), 270)
        self.add_port((-values['fork_width'] / 2 + values['bridge'],
                       values['base_length'] + values['bridge']), 90)
        self.add_port((+values['fork_width'] / 2 - values['bridge'],
                       values['base_length'] + values['bridge']), 90)
        self.add_port((0,
                       values['base_length'] + values['bridge']), 90)


class JJ_Yfork(EmptyGeometry):
    default_values = {
        'base_width': 5,
        'base_length': 20,

        'leg_length': 20,
        'fork_width': 20,
        'patch_offset': 3,

        'JJ_width1': 0.5,
        'JJ_width2': 0.3,

        "layer": 1,
    }

    def _drawing(self, values):
        self.name = 'JJ_Yfork'

        self + gdspy.Rectangle(
            (-values['base_width']/2, 0),
            (values['base_width']/2, values['base_length']),
            layer=values['layer']
        )

        self + gdspy.Rectangle(
            (-values['fork_width'] / 2, values['base_length'] - values['base_width']),
            (values['fork_width'] / 2, values['base_length']),
            layer=values['layer']
        )

        patch = gdspy.Polygon(
            [
                (-values['base_width']/2,
                 values['base_length'] - values['base_width']),
                (-values['fork_width']/2,
                 values['base_length'] - values['base_width']),
                (-values['base_width'] / 2,
                 values['base_length'] - values['base_width'] - (values['fork_width']/2 - values['base_width']/2)),
            ],
            layer=values['layer']
        )

        self + patch
        self + patch.mirror(self.position, self.position + np.array([0, 1]))


        self.add_port((-values['fork_width'] / 2 + values['base_width'] / 2,
                       values['base_length']), 90)

        self.add_port((+values['fork_width'] / 2 - values['base_width'] / 2,
                       values['base_length']), 90)

        leg1 = JJ_Ileg(base_width=values['base_width'],
                      length=values['leg_length'],
                      patch_offset=values['patch_offset'],
                      JJ_width=values['JJ_width1'],
                      layer=values['layer'])

        leg2 = JJ_Ileg(base_width=values['base_width'],
                      length=values['leg_length'],
                      patch_offset=values['patch_offset'],
                      JJ_width=values['JJ_width2'],
                      layer=values['layer'])

        leg1.merge_with(self.ports[0], 0)
        self + leg1
        del self.ports[0]

        leg2.merge_with(self.ports[0], 0)
        self + leg2
        del self.ports[0]


        self.add_port((0, 0), 270)
        self.add_port((0,
                       values['base_length'] + values['leg_length']), 90)


class DCsquid(EmptyGeometry):
    default_values = {
        'base_width': 2,
        'base_length_Y': 15,
        'base_length_T': 15,

        'fork_width': 15,
        'leg_length': 15,
        'patch_offset': 4,

        'JJ_width_Y1': 0.7,
        'JJ_width_Y2': 0.3,
        'JJ_width_T': 0.4,

        'bridge': 1,
        "layer": 1,

    }

    def _drawing(self, values):
        self.name = 'DCsquid'
        fork_Y = JJ_Yfork(
            base_width=values['base_width'],
            base_length=values['base_length_Y'],
            leg_length=values['leg_length'],
            fork_width=values['fork_width'],
            patch_offset=values['patch_offset'],
            JJ_width1=values['JJ_width_Y1'],
            JJ_width2=values['JJ_width_Y2'],
            layer=values['layer']
        )
        fork_T = JJ_Tfork(
            base_width=values['base_width'],
            base_length=values['base_length_T'],
            fork_width=values['fork_width'],
            patch_offset=values['patch_offset'],
            JJ_width=values['JJ_width_T'],
            bridge=values['bridge'],
            layer=values['layer'],
        )
        fork_T.merge_with(fork_Y.ports[1], 3)

        self + fork_Y
        self + fork_T

        self.ports += [fork_Y.ports[0]]
        self.ports += [fork_T.ports[0]]


class JJ_double(EmptyGeometry):
    default_values = {
        'base_width': 2,
        'length_I': 8,
        'base_length_L': 18,
        'leg_length_L': 5,
        'patch_offset': 2,

        'JJ_width_L': 0.5,
        'JJ_width_I': 0.5,

        'bridge': 1,
        "layer": 1,

    }

    def _drawing(self, values):
        self.name = 'JJ_double'
        leg_L = JJ_Lleg(
            base_width=values['base_width'],
            base_length=values['base_length_L'],
            leg_length=values['leg_length_L'],
            patch_offset=values['patch_offset'],
            JJ_width=values['JJ_width_L'],
            bridge=values['bridge'],
            layer=values['layer'],
        )

        leg_I = JJ_Ileg(
            base_width=values['base_width'],
            length=values['length_I']/2,
            patch_offset=values['patch_offset'],
            JJ_width=values['JJ_width_I'],
            layer=values['layer'],
        )
        self + leg_I
        leg_L.merge_with(leg_I.ports[1], 1)
        self + leg_L
        # self.ports += [leg_L.ports[0]]

        leg_I.merge_with(leg_I.ports[0], 0)
        self + leg_I
        leg_L.mirror('v')
        leg_L.merge_with(leg_I.ports[1], 1)
        self + leg_L
        self.ports += [leg_L.ports[0]]

        self.add_port((0, -values['length_I']/2-values['bridge']-values['base_length_L']), 270)


if __name__ == "__main__":
    # fork = JJ_Mfork()
    # fork = JJ_Ffork()
    # fork = JJ_leg()
    # fork = DCsquid()
    # fork = JJ_Lleg()
    fork = JJ_double()
    fork.translate((5, 0))
    fork.show()


