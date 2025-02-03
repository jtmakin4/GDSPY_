# import numpy as np
# import gdspy
# from matplotlib import pyplot as plt
# from collections.abc import Iterable
# import copy
# from .Settings import *
# from .Validator import *
# from .BaseFunctions import *
# from .BaseClasses import *
#
#
# class Geometry:
#     PORT_LAYER = Settings.PORT_LAYER
#     CROSS_CHECK_LAYER = Settings.CROSS_CHECK_LAYER
#
#     def __init__(self, name='Empty Geometry', unit=1e-6, precision=1e-9):
#         self.elements = [gdspy.PolygonSet([], layer=i) for i in range(0, 255)]
#         self.layers = [i for i in range(0, 255)]
#         self.isInverted = [False for i in range(0, 255)]  # Массив с информацией о том, является ли слой инвертированным
#
#         self.ports = []
#         self.curve_tolerance = 5
#         self.curve_precision = 1
#
#         self.name = name
#         self.unit = unit
#         self.precision = precision
#
#         self.position = (0, 0)
#         self.angle = 0
#
#         self.float = True  # флаг, включающий возможность манипулировать объектом
#
#     def move(self, dposition):
#         if self.float:
#             self.elements = [element.translate(dposition[0], dposition[1]) for element in self.elements]
#             for i in range(0, len(self.ports)):
#                 self.ports[i].translate(dposition)
#             self.position = (self.position[0] + dposition[0], self.position[1] + dposition[1])
#
#     def placeOn(self, position):
#         if self.float:
#             dposition = -np.array(self.position) + np.array(position)
#             self.elements = [element.translate(dposition[0], dposition[1]) for element in self.elements]
#             for i in range(0, len(self.ports)):
#                 self.ports[i].translate(dposition)
#             self.position = (self.position[0] + dposition[0], self.position[1] + dposition[1])
#
#     def rotate(self, dangle, center=None):
#         if self.float:
#             if center == None:
#                 center = self.position
#             else:
#                 rmatrix = np.array([[np.cos(dangle), -np.sin(dangle)], [np.sin(dangle), np.cos(dangle)]])
#                 self.position = tuple(np.dot(rmatrix, np.array(self.position) - np.array(center)) + np.array(center))
#
#             self.elements = [element.rotate(dangle, center=center) for element in self.elements]
#
#             dangle = round(dangle*180/np.pi, 1)/180*np.pi
#
#             for i in range(0, len(self.ports)):
#                 self.ports[i].rotate(dangle, center=center)
#
#             self.angle += dangle
#
#     def draw(self):
#         gdspy.current_library = gdspy.GdsLibrary(name='', unit=self.unit, precision=self.precision)
#         cell = gdspy.Cell(name=self.name)
#         cell.add(self.elements)
#         for i in range(0, len(self.ports)):
#             label = gdspy.Label(str(i), self.ports[i].position, "nw", self.ports[i].angle, layer=self.PORT_LAYER)
#             cell.add(label)
#             # port_path = gdspy.RobustPath((0, 0), 0, layer=self.PORT_LAYER)
#             a, o = self.ports[i].basis()
#             # print(self.ports[i].position)
#             # port_path.parametric(lambda u: np.array(self.ports[i].position) + a*u*3e3, width=lambda u: 1e3*(1-u))
#             port_path = gdspy.Polygon([np.array(self.ports[i].position) + 1e3*a,
#                                        np.array(self.ports[i].position) + 0.2e3*o,
#                                        np.array(self.ports[i].position) - 0.2e3*o], layer=self.PORT_LAYER)
#             cell.add(port_path)
#             del port_path
#
#         gdspy.LayoutViewer(library=None, cells=cell)
#         # cell.remove_polygons(lambda pts, layer, datatype: True)
#         del cell
#
#     def getPorts(self):
#         for i in range(0, len(self.ports)):
#             print('Порт {0:d}, координаты ({1:f}, {2:f}), угол {3:f}'.format(i,
#                                                                  self.ports[i].position[0],
#                                                                  self.ports[i].position[1],
#                                                                  self.ports[i].angle))
#         return self.ports
#
#     def addPort(self, position, angle):
#         if self.float:
#             if angle >= 0 and angle < 2*np.pi:
#                 self.ports.append(Port(position, angle))
#             else:
#                 raise ValueError('Angle must be between 0 and 2pi')
#
#     def mergeWithPort(self, aim_port, port_num):
#         if self.float:
#             try:
#                 self.rotate(aim_port.angle - self.ports[port_num].angle + np.pi)
#                 self.move((-self.ports[port_num].position[0] + aim_port.position[0],
#                            -self.ports[port_num].position[1] + aim_port.position[1]))
#             except AttributeError:
#                 AttributeError("Port number is incorrect")
#             except:
#                 pass  # todo проверка на то, что цель является портом
#             # todo сделать возможным соединение линиями, добавить класс линий. Добавить логику для линий на чертеже
#
#     def boolean(self, boolObj, operand, layer):
#         if self.float:
#         #todo сделать проверку на то, что boolObj это PlanarGeometry
#             self.elements[layer] = gdspy.boolean(self.elements[layer], boolObj, operand,
#                                                  layer=layer, precision=self.curve_precision)
#         # сглаживание
#         try:
#             self.elements[layer] = gdspy.boolean(self.elements[layer].polygons, None, 'or', max_points=0,
#                                              layer=layer, precision=self.curve_precision)
#         except AttributeError:
#             pass
#
#     def scale(self, factor=1e-3):
#         for i in self.layers:
#             self.elements[i].scale(factor, factor)
#
#
# class Sketch(Geometry):
#     def __init__(self, name, unit=1e-6, precision=1e-9, layers=None):
#         super().__init__()
#         del self.isInverted
#         del self.ports
#
#         self.portsLayer = Settings.PORT_LAYER
#         # todo добавить подстраиваемую точность, иначе будет много весить
#         self.curve_tolerance = 0.01
#         self.curve_precision = 0.01
#
#         self.geometries = []
#
#         self.unit = unit
#         self.precision = precision
#
#         self.name = name
#         self.lib = gdspy.GdsLibrary(name='library', unit=self.unit, precision=self.precision)
#         self.cell = self.lib.new_cell(self.name)
#
#
#     def setUnit(self, value):
#         self.unit = value
#
#     def setPrecision(self, value):
#         self.precision = value
#
#     def addGeometry(self, geometry):
#         geometry.float = False
#         self.geometries.append(geometry)
#         for i in geometry.layers:
#             if geometry.isInverted[i]:
#                 try:
#                     self.boolean(geometry.elements[i], 'not', i)
#                 except IndexError:
#                     pass
#             else:
#                 try:
#                     self.boolean(geometry.elements[i], 'or', i)
#                 except IndexError:
#                     pass
#
#     def fracture(self, max_points=199, precision=0.0001):
#         for i in range(0, len(self.elements)):
#             self.elements[i] = self.elements[i].fracture(max_points=max_points, precision=precision)
#
#     def draw(self, show=True, showPorts=False, fractured=True):
#         if fractured:
#             self.fracture()
#
#         gdspy.current_library = self.lib
#         self.cell.add(self.elements)
#
#         # рисование портов
#         if showPorts:
#             for geometry in self.geometries:
#                 for i in range(0, len(geometry.ports)):
#                     label = gdspy.Label(str(i), geometry.ports[i].position, "nw", geometry.ports[i].angle, layer=self.portsLayer)
#                     self.cell.add(label)
#                     a, o = geometry.ports[i].basis()
#
#                     port_path = gdspy.Polygon([np.array(geometry.ports[i].position) + 1e3 * a,
#                                                np.array(geometry.ports[i].position) + 0.2e3 * o,
#                                                np.array(geometry.ports[i].position) - 0.2e3 * o], layer=self.portsLayer)
#
#                     self.cell.add(port_path)
#
#         if show:
#             gdspy.LayoutViewer(pattern={'default': 0})
#
#     def clearLayer(self, layer_n):
#         self.cell.remove_polygons(lambda pts, layer, datatype: layer == layer_n)
#
#     def clearCell(self):
#         self.cell.remove_polygons(lambda pts, layer, datatype: True)
#
#     def refine(self):
#         for layer in range(0, 255):
#             try:
#                 working = self.cell.get_polygons(by_spec=True)[(layer, 0)]
#                 self.clearLayer(layer)
#                 result = gdspy.boolean(working, None, 'or', max_points=0, layer=layer)
#                 self.cell.add(result)
#             except KeyError:
#                 pass
#
#
#     def saveGDS(self, filename):
#         self.lib.write_gds(filename + '.gds')
#
#     def saveSVG(self, filename):
#         self.cell.write_svg(filename + '.svg')
