import gdspy
import numpy as np


class Port:
    """
    Объект Порт, являющийся дополнениям к PolygonSet пакета Numpy
    """
    def __init__(self, position, angle):
        if len(position) == 2 and type(position) == tuple:
            self.position = position
        else:
            raise TypeError('position is not a point')

        if type(angle) == float:
            self.angle = angle
        elif type(angle) == int:
            self.angle = float(angle)
        else:
            raise TypeError('angle is not float number')

    def translate(self, dposition):
        """
        Двигает порт вместе с объектом
        """
        if len(dposition) == 2 and type(dposition) == tuple:
            try:
                self.position = tuple([self.position[i] + dposition[i] for i in [0, 1]])
            except AttributeError:
                raise AttributeError("position is not a point")
        else:
            raise TypeError('displacement is not a point')

    def rotate(self, dangle, center):
        """
        Вращает порт вместе с объектом
        """
        if type(dangle) == float:
            self.angle += dangle
        else:
            raise TypeError('rotating angle is not float number')

        if len(center) == 2 and type(center) == tuple:
            try:
                r = np.array(list(self.position)) - np.array(list(center))
                self.position = np.array([[np.cos(dangle), -np.sin(dangle)],
                                          [np.sin(dangle), np.cos(dangle)]]).dot(r) + np.array(list(center))
                self.position = tuple(self.position)
            except AttributeError:
                raise AttributeError("position is not a point")
        else:
            raise TypeError('displacement is not a point')


class Geometry:
    def __init__(self, name='Empty Geometry', unit=1e-6, precision=1e-9,):
        self.elements = [gdspy.PolygonSet([], layer=i) for i in range(0, 255)]
        self.layers = [i for i in range(0, 255)]
        self.isInverted = [False for i in range(0, 255)] # Массив с информацией о том, является ли слой инвертированным
        self.portsLayer = 4
        self.ports = []

        self.name = name
        self.unit = unit
        self.precision = precision

        self.position = (0, 0)
        self.angle = 0

    def move(self, dposition):
        self.elements = [element.translate(dposition[0], dposition[1]) for element in self.elements]
        for i in range(0, len(self.ports)):
            self.ports[i].translate(dposition)
        self.position = (self.position[0] + dposition[0], self.position[1] + dposition[1])

    def rotate(self, dangle):
        self.elements = [element.rotate(dangle, center=self.position) for element in self.elements]
        for i in range(0, len(self.ports)):
            self.ports[i].rotate(dangle, center=self.position)
        self.angle += dangle

    def draw(self):
        gdspy.current_library = gdspy.GdsLibrary(name='', unit=self.unit, precision=self.precision)
        cell = gdspy.Cell(name=self.name)
        cell.add(self.elements)
        for i in range(0, len(self.ports)):
            label = gdspy.Label(str(i), self.ports[i].position, "nw", self.ports[i].angle, layer=self.portsLayer)
            cell.add(label)

        gdspy.LayoutViewer(library=None, cells=cell)
        del cell


    def getPorts(self):
        for i in range(0, len(self.ports)):
            print('Порт {0:d}, координаты ({1:f}, {2:f}), угол {3:f}'.format(i,
                                                                 self.ports[i].position[0],
                                                                 self.ports[i].position[1],
                                                                 self.ports[i].angle))
        return self.ports

    def addPort(self, position, angle):
        self.ports.append(Port(position, angle))

    def mergeWithPort(self, aim_port, port_num):
        try:
            self.rotate(self.ports[port_num].angle - aim_port.angle + np.pi)
            self.move((self.ports[port_num].position[0] - aim_port.position[0],
                       self.ports[port_num].position[1] - aim_port.position[1]))
        except AttributeError:
            AttributeError("Port number is incorrect")
        except:
            pass  # todo проверка на то, что цель является портом
        # todo сделать возможным соединение линиями, добавить класс линий. Добавить логику для линий на чертеже

    def boolean(self, boolObj, operand, layer):
        #todo сделать проверку на то, что boolObj это PlanarGeometry
        self.elements[layer] = gdspy.boolean(self.elements[layer], boolObj, operand, layer=layer)


# class PlanarGeometry:
#     def __init__(self, elements=[]):
#         self.gdspyGeometry = elements
#         # todo проверка на то, что геометрия это PolygonSet
#         self.ports = []
#         self.position = (0, 0)
#         self.name = 'PlanarGeometry'
#
#     def __call__(self, sketch):
#         # при вызове экземпляра класса происходит вывод объекта на экран так, как он бы выглядел в реальной модели с настройками sketch
#         # todo переиминовать sketch - это класс окружение, в котором существует модель
#
#         lib = gdspy.GdsLibrary(name='library', unit=sketch.unit, precision=sketch.precision)
#         cell = lib.new_cell(self.name)
#         cell.add(self.gdspyGeometry)
#         gdspy.LayoutViewer()
#         del(cell, lib)
#         return self.gdspyGeometry, self.ports
#
#     def translate(self, dx, dy):
#         self.gdspyGeometry = [_.translate(dx, dy) for _ in self.gdspyGeometry]
#         for i in range(0, len(self.ports)):
#             self.ports[i]._move((dx, dy))
#         self.position = (self.position[0] + dx, self.position[1] + dy)
#
#     def rotate(self, angle):
#         self.gdspyGeometry = [_.rotate(angle, center=self.position) for _ in self.gdspyGeometry]
#         for i in range(0, len(self.ports)):
#             self.ports[i]._rotate(angle, self.position)
#
#     def addPort(self, position, angle):
#         self.ports.append(Port(position, angle))
#
#     def getPorts(self):
#         return self.ports
#
#     def mergeWithPort(self, aim_port, port_num):
#         try:
#             self.rotate(self.ports[port_num].angle - aim_port.angle + np.pi)
#             self.translate(self.ports[port_num].position[0] - aim_port.position[0], self.ports[port_num].position[1] - aim_port.position[1])
#         except AttributeError:
#             AttributeError("Port number is incorrect")
#         except:
#             pass # todo проверка на то, что цель является портом
#         # todo сделать возможным соединение линиями, добавить класс линий. Добавить логику для линий на чертеже

class Mounting(Geometry):
    # по идее в подобных дочерних классах должна быть только их инициализация, остольные функции прописаны в классе-родителе. Возможно это можно сделать еще более просто.
    def __init__(self):
        super().__init__()

        # Рисование происходит здесь



        # записываем результат

        # задаем порты
        self.addPort(position=(0, 0), angle=0.)
