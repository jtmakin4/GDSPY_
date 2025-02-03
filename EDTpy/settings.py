# class Settings:
#
#     UNIT = 1e-6
#     PRECISION = 1e-9
#
#     COORDINATE_QUANT = PRECISION/UNIT
#     VALUE_QUANT = 0.001
#     ANGLE_QUANT = 0.001
#
#     MAX_POLYGON_POINTS = 2**13
#     DATATYPE = 0
#     NUMBER_OF_CIRCLE_POINTS = 100
#     CURVE_TOLERANCE = PRECISION / UNIT
#     CURVE_PRECISION = CURVE_TOLERANCE / 1
#
#     PORT_LAYER = 10
#     COVER_AREA_LAYER = 255

class Settings:

    UNIT = 1e-6
    PRECISION = 1e-9

    COORDINATE_QUANT = PRECISION/UNIT
    VALUE_QUANT = 1e-6
    ANGLE_QUANT = 1e-3
    PORT_LAYER = 10

    MAX_POLYGON_POINTS = 2**15
    DATATYPE = 0
    NUMBER_OF_CIRCLE_POINTS = 10
    CURVE_TOLERANCE = PRECISION / UNIT * 10
    CURVE_PRECISION = CURVE_TOLERANCE / 1

    # True - metal layer, False - substrate layer, None - others

    class Layer:
        def __init__(self, name, layer, is_metal):
            self.name = name
            self.layer = layer
            if is_metal in [True, False, None]:
                self.is_metal = is_metal
            else:
                raise ValueError("isMetal should be bool or None")

    LAYERS = {
        0: Layer('metal, 15+20*2 um', 0, True),
        1: Layer('RO4003C, 203 um', 1, False),
        2: Layer('metal, 15 um', 2, True),
        3: Layer('RO4450F(pepr.), 102 um', 3, False),
        4: Layer('metal, 15 um', 4, True),
        5: Layer('RO4003C, 203 um', 5, False),
        6: Layer('metal (TL), 15+20*2 um', 6, True),
        7: Layer('via,  200 um, through', 7, None),
    }
    VIA_LAYER = 7
    CPW_LAYER = 0
    VIA_D = 250 # диаметр дырок
    VIA_DIST = 700

