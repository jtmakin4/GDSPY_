class Settings:

    UNIT = 1e-6
    PRECISION = 1e-9

    COORDINATE_QUANT = PRECISION/UNIT
    ANGLE_QUANT = 5
    PORT_LAYER = 10
    CROSS_CHECK_LAYER = 11

    MAX_POLYGON_POINTS = 2**13

    def __init__(self):
        # по идее сюда надо вставлять пользовательские параметры
        pass