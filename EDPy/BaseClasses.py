import numpy as np
import gdspy
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

from .Settings import *


def extract(obj, layer, datatype=0, save_geom=True):
    """
    :param obj: Объект PolygonSet, NewGeometry из которого извлекаются все полигоны определенного слоя и типа данных
    :param layer: номер извлекаемого слоя
    :param datatype: тип данных извлекаемого слоя
    :param save_geom: сохранить ли содержимое исходного объекта
    :return: PolygonSet, содержащий извлеченные данные. Если данные отсутствуют, возвращает None
    """
    Validator.polygone_geometry(obj)
    mask = [obj.layers[i] != layer or obj.datatypes[i] != datatype for i in range(len(obj.polygons))]
    reversed_mask = [not mask[i] for i in range(len(mask))]

    if any(reversed_mask):
        result = gdspy.PolygonSet(list(np.array(obj.polygons)[reversed_mask]))
        result.layers = list(np.array(obj.layers)[reversed_mask])
        result.datatypes = list(np.array(obj.datatypes)[reversed_mask])
    else:
        result = None

    if not save_geom:
        obj.polygons = list(np.array(obj.polygons)[mask])
        obj.layers = list(np.array(obj.layers)[mask])
        obj.datatypes = list(np.array(obj.datatypes)[mask])

    return result


def append(base_obj,  tool_obj):
    Validator.polygone_geometry(base_obj)
    Validator.polygone_geometry(tool_obj)

    base_obj.polygons += tool_obj.polygons
    base_obj.layers += tool_obj.layers
    base_obj.datatypes += tool_obj.datatypes
    return base_obj


def find_layers_datatypes(obj):
    Validator.polygone_geometry(obj)
    return list(set(obj.layers)), list(set(obj.datatypes))


def boolean(base_obj, tool_obj, operation,
            precision=Settings.PRECISION/Settings.UNIT, max_points=Settings.MAX_POLYGON_POINTS):
    Validator.polygone_geometry(base_obj)
    Validator.polygone_geometry(tool_obj)
    layers = list(set(find_layers_datatypes(base_obj)[0] + find_layers_datatypes(tool_obj)[0]))
    datatypes = list(set(find_layers_datatypes(base_obj)[1] + find_layers_datatypes(tool_obj)[1]))

    for layer in layers:
        for datatype in datatypes:
            base = extract(base_obj, layer, datatype, save_geom=False)
            tool = extract(tool_obj, layer, datatype, save_geom=True)
            result = gdspy.boolean(base, tool, operation, layer=layer, datatype=datatype,
                                   precision=precision,
                                   max_points=max_points)
            if result is not None:
                append(base_obj, result)

    return base_obj


class Validator:

    @classmethod
    def polygone_geometry(cls, arg):
        if not(issubclass(arg.__class__, gdspy.PolygonSet) or issubclass(arg.__class__, NewGeometry)):
            raise TypeError('Argument should be PolygonSet or NewGeometry')

    @classmethod
    def point(cls, arg):
        if isinstance(arg, int) or isinstance(arg, float):
            # округление до кванта
            vector = round(arg/Settings.COORDINATE_QUANT) * Settings.COORDINATE_QUANT
            return vector
        else:
            raise TypeError('Argument should be int or float')


    @classmethod
    def vector_2d(cls, arg):
        if isinstance(arg, tuple) or isinstance(arg, list):
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

    # округление градусов до кванта
    @classmethod
    def quant_deg(cls, angle):
        return round(angle/Settings.ANGLE_QUANT) * Settings.ANGLE_QUANT


class Port(Validator):
    """
    Объект Порт, являющийся дополнениям к PolygonSet пакета Numpy. Угол направления ипорта дается в градусах и
    округляется до значения ANGLE_QUANT градусов. Угол указывается в пределах [0,360) градусов
    """

    __slots__ = ('__position', '__deg_angle', '__angle', '__a', '__o')

    def __new__(cls, *args, **kwargs):
        # print('... Creating  Port ...', end='\r')
        return super().__new__(cls)

    def __init__(self, position, angle):

        self.__position = self.vector_2d(position)
        self.__deg_angle = self.angle_quanted(angle)

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
        dangle_rad = dangle*np.pi/180.

        r = np.array(self.__position) - np.array(center)
        self.__position = np.array([[np.cos(dangle_rad), -np.sin(dangle_rad)],
                                    [np.sin(dangle_rad), np.cos(dangle_rad)]]).dot(r) + np.array(center)
        self.__position = self.vector_2d(tuple(self.__position))
        # self.__position = tuple(self.__position)

        # задаем новый базис и угол в радианах
        self.__angle = self.__deg_angle * np.pi/180.
        self.__a, self.__o = self.find_basis(self.__angle)

    # определение считываемых свойств
    @property
    def angle(self):
        return self.__deg_angle

    @angle.setter
    def angle(self, ang):
        ang = self.angle_quanted(ang)
        self.rotate(ang - self.__angle)

    @property
    def position(self):
        return self.__position

    @position.setter
    def position(self, vec):
        vec = self.vector_2d(vec)
        self.translate(np.array(vec) - np.array(self.__position))

    @property
    def basis(self):
        return self.__a, self.__o

    # построение ортонормированного базиса для порта
    @classmethod
    def find_basis(cls, angle):
        a = np.array([np.cos(angle), np.sin(angle)])
        o = np.array([np.cos(angle+np.pi/2), np.sin(angle+np.pi/2)])
        return tuple(a), tuple(o)


class Geometry:
    PORT_LAYER = Settings.PORT_LAYER
    CROSS_CHECK_LAYER = Settings.CROSS_CHECK_LAYER

    def __init__(self, name='Empty Geometry', unit=1e-6, precision=1e-9):
        self.elements = [gdspy.PolygonSet([], layer=i) for i in range(0, 255)]
        self.layers = [i for i in range(0, 255)]
        self.isInverted = [False for i in range(0, 255)] # Массив с информацией о том, является ли слой инвертированным

        self.ports = []
        self.curve_tolerance = 5
        self.curve_precision = 1

        self.name = name
        self.unit = unit
        self.precision = precision

        self.position = (0, 0)
        self.angle = 0

        self.float = True # флаг, включающий возможность манипулировать объектом

    def move(self, dposition):
        if self.float:
            self.elements = [element.translate(dposition[0], dposition[1]) for element in self.elements]
            for i in range(0, len(self.ports)):
                self.ports[i].translate(dposition)
            self.position = (self.position[0] + dposition[0], self.position[1] + dposition[1])

    def placeOn(self, position):
        if self.float:
            dposition = -np.array(self.position) + np.array(position)
            self.elements = [element.translate(dposition[0], dposition[1]) for element in self.elements]
            for i in range(0, len(self.ports)):
                self.ports[i].translate(dposition)
            self.position = (self.position[0] + dposition[0], self.position[1] + dposition[1])

    def rotate(self, dangle, center=None):
        if self.float:
            if center == None:
                center = self.position
            else:
                rmatrix = np.array([[np.cos(dangle), -np.sin(dangle)], [np.sin(dangle), np.cos(dangle)]])
                self.position = tuple(np.dot(rmatrix, np.array(self.position) - np.array(center)) + np.array(center))

            self.elements = [element.rotate(dangle, center=center) for element in self.elements]

            dangle = round(dangle*180/np.pi, 1)/180*np.pi

            for i in range(0, len(self.ports)):
                self.ports[i].rotate(dangle, center=center)

            self.angle += dangle

    def draw(self):
        gdspy.current_library = gdspy.GdsLibrary(name='', unit=self.unit, precision=self.precision)
        cell = gdspy.Cell(name=self.name)
        cell.add(self.elements)
        for i in range(0, len(self.ports)):
            label = gdspy.Label(str(i), self.ports[i].position, "nw", self.ports[i].angle, layer=self.PORT_LAYER)
            cell.add(label)
            # port_path = gdspy.RobustPath((0, 0), 0, layer=self.PORT_LAYER)
            a, o = self.ports[i].basis()
            # print(self.ports[i].position)
            # port_path.parametric(lambda u: np.array(self.ports[i].position) + a*u*3e3, width=lambda u: 1e3*(1-u))
            port_path = gdspy.Polygon([np.array(self.ports[i].position) + 1e3*a,
                                       np.array(self.ports[i].position) + 0.2e3*o,
                                       np.array(self.ports[i].position) - 0.2e3*o], layer=self.PORT_LAYER)
            cell.add(port_path)
            del port_path

        gdspy.LayoutViewer(library=None, cells=cell)
        # cell.remove_polygons(lambda pts, layer, datatype: True)
        del cell

    def getPorts(self):
        for i in range(0, len(self.ports)):
            print('Порт {0:d}, координаты ({1:f}, {2:f}), угол {3:f}'.format(i,
                                                                 self.ports[i].position[0],
                                                                 self.ports[i].position[1],
                                                                 self.ports[i].angle))
        return self.ports

    def addPort(self, position, angle):
        if self.float:
            if angle >= 0 and angle < 2*np.pi:
                self.ports.append(Port(position, angle))
            else:
                raise ValueError('Angle must be between 0 and 2pi')

    def mergeWithPort(self, aim_port, port_num):
        if self.float:
            try:
                self.rotate(aim_port.angle - self.ports[port_num].angle + np.pi)
                self.move((-self.ports[port_num].position[0] + aim_port.position[0],
                           -self.ports[port_num].position[1] + aim_port.position[1]))
            except AttributeError:
                AttributeError("Port number is incorrect")
            except:
                pass  # todo проверка на то, что цель является портом
            # todo сделать возможным соединение линиями, добавить класс линий. Добавить логику для линий на чертеже

    def boolean(self, boolObj, operand, layer):
        if self.float:
        #todo сделать проверку на то, что boolObj это PlanarGeometry
            self.elements[layer] = gdspy.boolean(self.elements[layer], boolObj, operand,
                                                 layer=layer, precision=self.curve_precision)
        # сглаживание
        try:
            self.elements[layer] = gdspy.boolean(self.elements[layer].polygons, None, 'or', max_points=0,
                                             layer=layer, precision=self.curve_precision)
        except AttributeError:
            pass

    def scale(self, factor=1e-3):
        for i in self.layers:
            self.elements[i].scale(factor, factor)


class Sketch(Geometry):
    def __init__(self, name, unit=1e-6, precision=1e-9, layers=None):
        super().__init__()
        del self.isInverted
        del self.ports

        self.portsLayer = Settings.PORT_LAYER
        # todo добавить подстраиваемую точность, иначе будет много весить
        self.curve_tolerance = 0.01
        self.curve_precision = 0.01

        self.geometries = []

        self.unit = unit
        self.precision = precision

        self.name = name
        self.lib = gdspy.GdsLibrary(name='library', unit=self.unit, precision=self.precision)
        self.cell = self.lib.new_cell(self.name)


    def setUnit(self, value):
        self.unit = value

    def setPrecision(self, value):
        self.precision = value

    def addGeometry(self, geometry):
        geometry.float = False
        self.geometries.append(geometry)
        for i in geometry.layers:
            if geometry.isInverted[i]:
                try:
                    self.boolean(geometry.elements[i], 'not', i)
                except IndexError:
                    pass
            else:
                try:
                    self.boolean(geometry.elements[i], 'or', i)
                except IndexError:
                    pass

    def fracture(self, max_points=199, precision=0.0001):
        for i in range(0, len(self.elements)):
            self.elements[i] = self.elements[i].fracture(max_points=max_points, precision=precision)

    def draw(self, show=True, showPorts=False, fractured=True):
        if fractured:
            self.fracture()

        gdspy.current_library = self.lib
        self.cell.add(self.elements)

        # рисование портов
        if showPorts:
            for geometry in self.geometries:
                for i in range(0, len(geometry.ports)):
                    label = gdspy.Label(str(i), geometry.ports[i].position, "nw", geometry.ports[i].angle, layer=self.portsLayer)
                    self.cell.add(label)
                    a, o = geometry.ports[i].basis()

                    port_path = gdspy.Polygon([np.array(geometry.ports[i].position) + 1e3 * a,
                                               np.array(geometry.ports[i].position) + 0.2e3 * o,
                                               np.array(geometry.ports[i].position) - 0.2e3 * o], layer=self.portsLayer)

                    self.cell.add(port_path)

        if show:
            gdspy.LayoutViewer(pattern={'default': 0})

    def clearLayer(self, layer_n):
        self.cell.remove_polygons(lambda pts, layer, datatype: layer == layer_n)

    def clearCell(self):
        self.cell.remove_polygons(lambda pts, layer, datatype: True)

    def refine(self):
        for layer in range(0, 255):
            try:
                working = self.cell.get_polygons(by_spec=True)[(layer, 0)]
                self.clearLayer(layer)
                result = gdspy.boolean(working, None, 'or', max_points=0, layer=layer)
                self.cell.add(result)
            except KeyError:
                pass


    def saveGDS(self, filename):
        self.lib.write_gds(filename + '.gds')

    def saveSVG(self, filename):
        self.cell.write_svg(filename + '.svg')


# нужно ли исправлять GDSPY версию сеттов полигонов и булевых операций над ними?
class NewGeometry(gdspy.PolygonSet, Validator):
    CURVE_PRECISION = Settings.PRECISION/Settings.UNIT
    CURVE_TOLERANCE = CURVE_PRECISION

    __slots__ = ('isInverted', 'name', '__position', '__angle', 'ports')

    def __init__(self, name='Empty Geometry', position=(0, 0), angle=0):
        super().__init__([])
        self.isInverted = [False] * 255 # Массив с информацией о том, является ли слой инвертированным. Нужен только при добавлении в скетч
        self.ports = []
        self.name = str(name)
        self.__position = self.vector_2d(position)
        self.__angle = self.angle_quanted(angle)

    # еще раз пройдись на свежую голову по свойствам и пеермещениям, везде расставь проверки!

    def translate(self, dposition):
        dposition = self.vector_2d(dposition)

        super().translate(*tuple(dposition))
        for i in range(0, len(self.ports)):
            self.ports[i].translate(dposition)
        self.__position = tuple(np.array(self.__position) + np.array(dposition))

    def rotate(self, angle, center=None):
        if center is None:
            center = self.__position

        center = self.vector_2d(center)
        angle = self.angle_quanted(angle)

        for i in range(0, len(self.ports)):
            self.ports[i].rotate(angle, center)
        self.__angle = self.angle_quanted(self.__angle + angle)

        angle_rad = angle * np.pi / 180
        r = np.array(self.__position) - np.array(center)
        self.__position = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad), np.cos(angle_rad)]]).dot(r) + np.array(center)
        self.__position = tuple(self.__position)
        super().rotate(angle_rad, center)

    def merge_with(self, aim_port, port_num):
        if isinstance(aim_port, Port):
            if 0 <= port_num < len(self.ports):
                self.rotate(aim_port.angle - self.ports[port_num].angle + 180)
                self.translate((-self.ports[port_num].position[0] + aim_port.position[0],
                                -self.ports[port_num].position[1] + aim_port.position[1]))
            else:
                raise ValueError("Geometry don't have port with this number")
        else:
            raise TypeError('aim_port should have Port type')

    def set_port(self, position, angle):
        position = self.vector_2d(position)
        angle = self.angle_quanted(angle)
        self.ports += [Port(position, angle)]

    @property
    def angle(self):
        return self.__angle

    @angle.setter
    def angle(self, ang):
        ang = self.angle_quanted(ang)
        self.rotate(ang - self.__angle)

    @property
    def position(self):
        return self.__position

    @position.setter
    def position(self, vec):
        vec = self.vector_2d(vec)
        self.translate(np.array(vec) - np.array(self.__position))

    # теперь при вызове будет рисоваться чертеж
    def __repr__(self):
        gdspy.current_library = gdspy.GdsLibrary(name='', unit=Settings.UNIT, precision=Settings.PRECISION)
        cell = gdspy.Cell(name=self.name)
        cell.add(self)

        # вспомогательные элементы
        bonding_box = self.get_bounding_box()
        width = bonding_box[1, 0] - bonding_box[0, 0]
        height = bonding_box[1, 1] - bonding_box[0, 1]
        support_elements_size = round(min(width, height)/5)
        # print(type(support_elements_size))

        # центр объекта
        center_point = gdspy.Round(self.__position, support_elements_size - support_elements_size/10,
                                                 initial_angle=np.pi,
                                                 final_angle=3*np.pi/2,
                                   layer=Settings.PORT_LAYER)

        center_point = gdspy.boolean(center_point,
                                     gdspy.Round(self.__position, support_elements_size - support_elements_size/10,
                                                 initial_angle=0,
                                                 final_angle=np.pi/2), 'or', layer=Settings.PORT_LAYER)
        center_point.rotate(self.__angle*np.pi/180, self.__position)
        cell.add(center_point)

        # центр координат
        center_point = gdspy.Round((0, 0), support_elements_size,
                                   inner_radius=support_elements_size - support_elements_size/10,
                                   layer=Settings.PORT_LAYER)
        cell.add(center_point)
        center_point = gdspy.Round((0, 0), support_elements_size/5,
                                   layer=Settings.PORT_LAYER)
        cell.add(center_point)

        for i in range(0, len(self.ports)):
            label = gdspy.Label(str(i), self.ports[i].position, "nw", self.ports[i].angle, layer=Settings.PORT_LAYER)
            cell.add(label)

            a, o = self.ports[i].basis

            port_path = gdspy.Polygon([np.array(self.ports[i].position) + support_elements_size*np.array(a),
                                       np.array(self.ports[i].position) + support_elements_size*np.array(o)/5,
                                       np.array(self.ports[i].position) - support_elements_size*np.array(o)/5], layer=Settings.PORT_LAYER)
            cell.add(port_path)

        gdspy.LayoutViewer(library=None, cells=cell)
        del cell
        return self.__str__()

    def __str__(self):
        return f'Object {self.name}, {self.__class__}\n' \
               f'position = {self.position}\n' \
               f'angle = {self.angle}\n' \
               f'Number of Ports:{len(self.ports)}'

    def boolean(self, tool_obj, operation,
                precision=Settings.PRECISION/Settings.UNIT, max_points=Settings.MAX_POLYGON_POINTS):
        return boolean(self, tool_obj, operation,
                       precision=precision, max_points=max_points)

    def __mul__(self, tool_obj):
        return self.boolean(tool_obj, 'and')

    def __sub__(self, tool_obj):
        return self.boolean(tool_obj, 'not')

    def __add__(self, tool_obj):
        return self.boolean(tool_obj, 'or')

    def __truediv__(self, tool_obj):
        return self.boolean(tool_obj, 'xor')

    def append(self, tool_obj):
        return append(self, tool_obj)


class Path(Validator):

    __slots__ = ('__path', '__spots', '__radiuses', '__max_radius', '__center_vectors', '__length')

    def __init__(self, path=None, radiuses=None):
        # инит содержит информацию только непосредственно о генерируемой кривой

        self.__path = path
        self.__radiuses = radiuses
        self.__max_radius = np.inf

        # векторы, необходимые чтобы из одной из вершин прийти к точке с единичным радиусом касательной
        self.__center_vectors = None

        # Споты(порты) для размещения элементов вроде мостов, via holes и тд.
        # Фактически споты содержат всю информацию о вспомогательных элементах
        # Споты не задаются при инициализации, для этого сущетсвуют отдельные методы. Это сделано для того, чтобы разгрузить код
        # В сами методы можно также добавить ссылки на объекты для совмещения с ними
        # Добавлять споты вручную нельзя
        self.__spots = []
        self.__length = None

    @ property
    def spots(self):
        return self.__spots


    # Старые функции для нахождения положения сквозных отверстий
    # Здесь не обработан случай, когда кривая состтоит только из окружностей
    # Для обхода этого Радиус уменьшен на 1 %
    @classmethod
    def _circular_bend_path(cls, input_array, bend_r):

        def _unit_vector(point1, point2):
            vec = np.array(point2) - np.array(point1)
            vec = vec / np.linalg.norm(vec)
            return vec

        def _array_simplify(input_array):
            # удаляет случайные лишние точки прямой
            array = np.array(input_array[0])
            for i in range(1, len(input_array) - 1):
                vec1 = _unit_vector(input_array[i - 1], input_array[i])
                vec2 = _unit_vector(input_array[i], input_array[i + 1])
                if vec1.dot(vec2) != 1:
                    array = np.vstack([array, input_array[i]])
            array = np.vstack([array, input_array[-1]])
            return array

        array = np.array(input_array[0])
        arc_centers = np.array([0, 0])
        angles = np.array([])
        for i in range(1, len(input_array) - 1):
            vec1 = _unit_vector(input_array[i - 1], input_array[i])
            vec2 = _unit_vector(input_array[i], input_array[i + 1])
            cos = vec1.dot(vec2)
            if cos != 1:
                alpha = np.arccos(cos)
                a = bend_r / np.tan(np.pi / 2 - alpha / 2)
                new_point1 = -vec1 * a + input_array[i]
                new_point2 = vec2 * a + input_array[i]
                array = np.vstack([array, new_point1, new_point2])

                angles = np.append(angles, alpha)
                if np.cross(vec1, vec2) < 0:
                    arc_centers = np.vstack([arc_centers, np.array([vec1[1], -vec1[0]]) * bend_r + new_point1])
                else:
                    arc_centers = np.vstack([arc_centers, np.array([-vec1[1], vec1[0]]) * bend_r + new_point1])
            else:
                array = np.vstack([array, input_array[i]])
        array = np.vstack([array, input_array[-1]])
        array = _array_simplify(array)
        return array, arc_centers[1:], angles

    @classmethod
    def _point_to_len_array(cls, points, angles, bend_r):
        length_array = np.array([0])
        for i in range(1, len(points)):
            if i % 2 == 1:
                vec = np.array(points[i] - points[i - 1])
                length_array = np.append(length_array, np.linalg.norm(vec) + length_array[-1])
            else:
                length_array = np.append(length_array, bend_r * angles[i // 2 - 1] + length_array[-1])
        return length_array[1:]

    @classmethod
    def _len_to_position(cls, l, points, centers, angles, bend_r):

        def _rotate_around(point, center, angle, diraction):
            if diraction > 0:
                rotate_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                          [np.sin(angle), np.cos(angle)]])
            else:
                rotate_matrix = np.array([[np.cos(angle), np.sin(angle)],
                                          [-np.sin(angle), np.cos(angle)]])
            return np.dot(rotate_matrix, point - center) + center

        length_array = cls._point_to_len_array(points, angles, bend_r)
        section_n = -1
        if l < length_array[0]:
            section_n = 0
        for i in range(1, len(length_array)):
            if l <= length_array[i] and l > length_array[i - 1]:
                section_n = i
        if section_n == -1:
            print("length is too big")
            return -1

        if section_n % 2 == 0:
            vec = points[section_n + 1] - points[section_n]
            vec = vec / np.linalg.norm(vec)
            result_point = points[section_n + 1] + vec * (l - length_array[section_n])
            return result_point, np.dot(vec, np.array([[0, -1], [1, 0]]))
        else:
            vec1 = points[section_n] - points[section_n - 1]
            vec2 = points[section_n + 2] - points[section_n + 1]
            diraction = np.cross(vec1, vec2)
            point_angle = (l - length_array[section_n - 1]) /\
                          (length_array[section_n] - length_array[section_n - 1]) * angles[section_n // 2]
            result_point = _rotate_around(points[section_n], centers[section_n // 2], point_angle, diraction)
            vec = np.array(result_point - centers[section_n // 2]) / np.linalg.norm(
                result_point - centers[section_n // 2])
            if diraction > 0:
                return result_point, vec
            else:
                return result_point, -vec

    @classmethod
    def _create_via_centers(cls, path, bend_radius, step, offset=0):
        path_array, bend_centers, bend_angles = cls._circular_bend_path(path, bend_radius)
        length_arr = cls._point_to_len_array(path_array, bend_angles, bend_radius)
        l_arr = np.arange(0, length_arr[-1], step)

        points_array_left = np.array([0, 0])
        points_array_right = np.array([0, 0])
        points_array = np.array([0, 0])

        for l in l_arr:
            p, vec = cls._len_to_position(l, path_array, bend_centers, bend_angles, bend_radius)
            points_array_left = np.vstack([points_array_left, p + vec * offset])
            points_array_right = np.vstack([points_array_right, p - vec * offset])
            points_array = np.vstack([points_array, p])

        if offset != 0:
            return [points_array_left[1:], points_array_right[1:]]
        else:
            return points_array


class AutoPath(Path, Validator):
    def __init__(self, port1, port2):
        super().__init__(self.get_trajectory(port1, port2))



    @classmethod
    def get_trajectory(cls, port1, port2, draw=False):
        # можно сделать декоратором!
        def draw_path():
            if draw:
                x_path = [path[i][0] for i in range(len(path))]
                y_path = [path[i][1] for i in range(len(path))]
                min_len = min((max(x_path) - min(x_path)), (max(y_path) - min(y_path)))
                max_len = max((max(x_path) - min(x_path)), (max(y_path) - min(y_path)))
                fig, ax = plt.subplots()
                ax.plot(x_path, y_path, '-.', marker='o', markersize=3, color="blue")

                if rad_vectors is not None:
                    # todo переделать отображение радиусов
                    # center_points = [path[i+1]+rad_vectors[i]*min_len/20*j
                    #                  for i in range(len(rad_vectors)) for j in range(10)]
                    # for point in center_points:
                    #     ax.plot([point[0]], [point[1]], marker='o', markersize=3, color="red")

                    center_points = [path[i + 1] + rad_vectors[i] * max_radius
                                     for i in range(len(rad_vectors))]

                    for point in center_points:
                        circle = plt.Circle(tuple(point), max_radius, fill=False)
                        ax.add_patch(circle)

                ax.axis('equal')
                ax.grid(True)
                plt.show()

        if not isinstance(port1, Port):
            raise TypeError('port1 should be Port type')

        if not isinstance(port2, Port):
            raise TypeError('port2 should be Port type')

        s1 = None
        s2 = None

        a1, o1 = port1.basis
        a1 = np.array(a1)
        o1 = np.array(o1)
        r1 = np.array(port1.position)

        a2, o2 = port2.basis
        a2 = np.array(a2)
        o2 = np.array(o2)
        r2 = np.array(port2.position)

        delt_r = r1 - r2

        # возвращаемые значения
        path = [r1, r2]
        rad_vectors = None #векторы, направленные на центр окружности из вершин
                           # модуль соответствует удалению от вершины для единичного касательного радиуса.
                           # по факту он равен 1/sin(угол биссектриссы угла)
        max_radius = None

        # плохое условеи на то, что порты смотрят в одну сторону. Иногда и при этом может построить, а иногда не защищает от поломки
        # if np.dot(a1, a2) > 0:
        #     print('Cannot draw path, ports have same diraction')
        #     return None, None

        # Опрределяем направление на центр окружностей (s - sign знак для вектора o портов)
        zero_value = 1e-10
        if np.dot(-delt_r, o1) >= zero_value:
            s1 = 1
        if np.dot(-delt_r, o1) <= zero_value:
            s1 = -1
        if abs(np.dot(delt_r, o2)) < zero_value:
            s1 = 0

        if np.dot(delt_r, o2) >= zero_value:
            s2 = 1
        if np.dot(delt_r, o2) <= zero_value:
            s2 = -1
        if abs(np.dot(delt_r, o2)) < zero_value:
            s2 = 0

        """
        Далее рассмотрены различные конфигурации линий
        """

        # Порты находятся на одной линии
        if s1 == 0 and s2 == 0:
            path = [r1,  r2]

            rad_vectors = None

            max_radius = None
            draw_path()
            return path, rad_vectors, max_radius

        # Один из портов смотрит на другой
        if (s1 == 0 and s2 != 0) or (s1 != 0 and s2 == 0):
            if s1 == 0 and s2 != 0:
                s1 = 1
            else:
                s2 = 1

            delt_o = s1 * o1 - s2 * o2
            D = np.dot(delt_o, delt_r) ** 2
            D += 2 * np.dot(delt_r, delt_r) * (1 + s1 * s2 * np.dot(o1, o2))
            max_radius = np.dot(delt_o, delt_r) + np.sqrt(D)
            max_radius /= (2 * (1 + s1 * s2 * np.dot(o1, o2)))

            c1 = r1 + max_radius * s1 * o1
            c2 = r2 + max_radius * s2 * o2
            delt_c = c1 - c2
            p = (c1 + c2) / 2

            # Создаем промужуточные точки на кривой
            x1 = np.dot(p - r1, delt_c) / np.dot(a1, delt_c)
            x2 = np.dot(p - r2, delt_c) / np.dot(a2, delt_c)

            # определяем направления на радиусы окружностей
            rad_dir_1 = (r1 + a1 * x1 - c1)/np.linalg.norm(r1 + a1 * x1 - c1)
            rad_dir_2 = (r2 + a2 * x2 - c2) / np.linalg.norm(r2 + a2 * x2 - c2)

            ang_cos1 = np.dot(-rad_dir_1, -a1) / np.linalg.norm(rad_dir_1) / np.linalg.norm(a1)
            ang_cos2 = np.dot(-rad_dir_2, -a2) / np.linalg.norm(rad_dir_2) / np.linalg.norm(a2)

            ang1 = np.arccos(ang_cos1)
            ang2 = np.arccos(ang_cos2)

            rad_vectors = [-rad_dir_1/np.sin(ang1), -rad_dir_2/np.sin(ang2)]

            path = [r1, r1 + a1 * x1, r2 + a2 * x2, r2]
            draw_path()
            return path, rad_vectors, max_radius

        # Порты друг на друга не смотрят, лучи, исходящие из портов, не пересекаются
        if s1*s2 > 0:
            if (1 + s1 * s2 * np.dot(o1, o2)) > 1e-9:
                delt_o = s1 * o1 - s2 * o2
                D = np.dot(delt_o, delt_r) ** 2
                D += 2 * np.dot(delt_r, delt_r) * (1 + s1 * s2 * np.dot(o1, o2))
                max_radius = np.dot(delt_o, delt_r) + np.sqrt(D)
                max_radius /= (2 * (1 + s1 * s2 * np.dot(o1, o2)))

                c1 = r1 + max_radius * s1 * o1
                c2 = r2 + max_radius * s2 * o2
                delt_c = c1 - c2
                p = (c1 + c2) / 2
                # Создаем промужуточные точки на кривой
                x1 = np.dot(p - r1, delt_c) / np.dot(a1, delt_c)
                x2 = np.dot(p - r2, delt_c) / np.dot(a2, delt_c)

                # определяем направления на радиусы окружностей
                rad_dir_1 = (r1 + a1 * x1 - c1) / np.linalg.norm(r1 + a1 * x1 - c1)
                rad_dir_2 = (r2 + a2 * x2 - c2) / np.linalg.norm(r2 + a2 * x2 - c2)

                # по идее это можно определить через доп функцию
                ang_cos1 = np.dot(-rad_dir_1, -a1) / np.linalg.norm(rad_dir_1) / np.linalg.norm(a1)
                ang_cos2 = np.dot(-rad_dir_2, -a2) / np.linalg.norm(rad_dir_2) / np.linalg.norm(a2)

                ang1 = np.arccos(ang_cos1)
                ang2 = np.arccos(ang_cos2)

                rad_vectors = [-rad_dir_1/np.sin(ang1), -rad_dir_2/np.sin(ang2)]

                path = [r1, r1 + a1 * x1, r2 + a2 * x2, r2]
                draw_path()
                return path, rad_vectors, max_radius
            else:
                # Направление портов параллельно
                angle = np.arccos(np.dot(delt_r, a2)/np.sqrt(np.dot(delt_r, delt_r)))
                t = np.sqrt(np.dot(delt_r, delt_r))/4/np.cos(angle)
                max_radius = t/np.tan(angle)

                c1 = r1 + max_radius * s1 * o1
                c2 = r2 + max_radius * s2 * o2

                rad_dir_1 = (r1 + a1 * t - c1) / np.linalg.norm(r1 + a1 * t - c1)
                rad_dir_2 = (r2 + a2 * t - c2) / np.linalg.norm(r2 + a2 * t - c2)

                ang_cos1 = np.dot(-rad_dir_1, -a1) / np.linalg.norm(rad_dir_1) / np.linalg.norm(a1)
                ang_cos2 = np.dot(-rad_dir_2, -a2) / np.linalg.norm(rad_dir_2) / np.linalg.norm(a2)

                ang1 = np.arccos(ang_cos1)
                ang2 = np.arccos(ang_cos2)

                rad_vectors = [-rad_dir_1/np.sin(ang1), -rad_dir_2/np.sin(ang2)]

                path = [r1, r1 + a1 * t, r2 + a2 * t, r2]
                draw_path()
                return path, rad_vectors, max_radius

        # Порты друг на друга не смотрят, лучи, исходящие из портов, пересекаются между портами (3 точки)
        if s1*s2 < 0:
            A = np.vstack((-a1, a2))
            t1, t2 = np.linalg.solve(A.transpose(), delt_r)
            r3 = r1+a1*t1 # точка пересечения прямых, выходящих из портов
            a3 = -a1-a2
            a3 = a3/np.sqrt(np.dot(a3, a3))  # биссектрисса

            ang_cos = np.dot(-a1, a3) / np.linalg.norm(a1) / np.linalg.norm(a3)
            ang = np.arccos(ang_cos)
            max_radius = min(t1, t2)*np.tan(ang)

            rad_vectors = [a3/np.sin(ang)]
            path = [r1, r3, r2]
            draw_path()
            return path, rad_vectors, max_radius

