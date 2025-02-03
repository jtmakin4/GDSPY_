from EDTpy.validator import Validator
from EDTpy.settings import Settings
import numpy as np
import gdspy
import copy


# todo сделать возможность отключать округление. Убрать округления для спотов
class Port(Validator, Settings):
    """
    Объект Порт, являющийся дополнениям к PolygonSet пакета Numpy. Угол направления ипорта дается в градусах и
    округляется до значения ANGLE_QUANT градусов. Угол указывается в пределах [0,360) градусов
    """

    __slots__ = ('__position', '__deg_angle', '__angle', '__a', '__o')

    def __init__(self, position, deg_angle):

        self.__position = self.vector_2d(position)
        self.__deg_angle = self.angle_quanted(deg_angle)

        # угол в радианах для нужд алгоритмов
        self.__angle = self.__deg_angle*np.pi/180

        self.__a, self.__o = self.find_basis(self.__angle)

    # отображение
    def __repr__(self):
        return f"{self.__class__}\nPort:\nposition: {self.__position}\nangle: {self.__deg_angle}\n" \
               f"basis: \na: {self.__a}\no: {self.__o}"

    def __str__(self):
        return f"position: {self.__position}\nangle: {self.__deg_angle}"

    def translate(self, dposition):
        """
        Двигает порт вместе с объектом
        """
        dposition = self.vector_2d(dposition)
        self.__position = tuple(np.array(self.__position) + np.array(dposition))
        return self

    def rotate(self, dangle, center=None):
        """
        Вращает порт вместе с объектом
        """
        # проверяем аргументы
        if center is not None:
            center = self.vector_2d(center)
        else:
            # None означает вращение вокруг своей оси
            center = self.__position

        # квантуем угол
        dangle = self.angle_quanted(dangle)

        # округляем, чтобы был в пределах [0, 360)
        self.__deg_angle += dangle
        if self.__deg_angle >= 360 or self.__deg_angle < 0:
            self.__deg_angle -= (self.__deg_angle//360) * 360

        # задаем новое положение центра

        dangle_rad = dangle * np.pi / 180.
        r = np.array(self.__position) - np.array(center)
        self.__position = np.array([[np.cos(dangle_rad), -np.sin(dangle_rad)],
                                    [np.sin(dangle_rad), np.cos(dangle_rad)]]).dot(r) + np.array(center)
        self.__position = self.vector_2d(tuple(self.__position))
        # self.__position = tuple(self.__position)

        # задаем новый базис и угол в радианах
        self.__angle = self.__deg_angle * np.pi/180.
        self.__a, self.__o = self.find_basis(self.__angle)
        return self

    # определение считываемых свойств
    @property
    def angle(self):
        return self.__deg_angle

    @angle.setter
    def angle(self, ang):
        ang = self.angle_quanted(ang)
        self.rotate(ang - self.__deg_angle)

    @property
    def position(self):
        return self.__position

    @position.setter
    def position(self, vec):
        vec = self.vector_2d(vec)
        self.translate(np.array(vec) - np.array(self.__position))

    @property
    def basis(self):
        return np.array(self.__a), np.array(self.__o)

    # построение ортонормированного базиса для порта
    @classmethod
    def find_basis(cls, angle):
        a = np.array([np.cos(angle), np.sin(angle)])
        o = np.array([np.cos(angle+np.pi/2), np.sin(angle+np.pi/2)])
        return tuple(a), tuple(o)
