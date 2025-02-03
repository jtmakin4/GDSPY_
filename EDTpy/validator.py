import gdspy
from EDTpy.settings import *
import numpy as np
from collections.abc import Sized, Iterable
from typing import Union as Union


class Validator:
    @classmethod
    def polygone_geometry(cls, arg):
        if not(issubclass(arg.__class__, gdspy.PolygonSet)):
            raise TypeError('Argument should be PolygonSet or NewGeometry')

    @classmethod
    def coordinate(cls, arg):
        if isinstance(arg, int) or isinstance(arg, float):
            # округление до кванта
            vector = round(arg/Settings.COORDINATE_QUANT) * Settings.COORDINATE_QUANT
            return vector
        else:
            raise TypeError('Argument should be int or float')

    @classmethod
    def vector_2d(cls, arg):
        if isinstance(arg, Iterable):
            if len(arg) != 2:
                raise ValueError('Argument should be 2-dim.')
        else:
            raise TypeError('Argument should be tuple or list')

        # округление до кванта
        vector = list(map(round, np.array(arg)/Settings.COORDINATE_QUANT))
        return tuple(np.array(vector) * Settings.COORDINATE_QUANT)

    @classmethod
    def angle_quanted(cls, arg):
        if not(isinstance(arg, int) or isinstance(arg, float)):
            raise TypeError('angle should be integer or float')
        # return cls.quant_deg(arg - (arg//360)*360)
        return round((arg - (arg//360)*360) / Settings.ANGLE_QUANT) * Settings.ANGLE_QUANT

    @classmethod
    def value_quanted(cls, arg):
        if isinstance(arg, int) or isinstance(arg, float):
            # округление до кванта
            vector = round(arg/Settings.VALUE_QUANT) * Settings.VALUE_QUANT
            return vector
        else:
            raise TypeError('Argument should be int or float')

    # # округление градусов до кванта
    # @classmethod
    # def quant_deg(cls, angle):
    #     return round(angle/Settings.ANGLE_QUANT) * Settings.ANGLE_QUANT
