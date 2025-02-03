from EDTpy.settings import *
from EDTpy.port import *
from EDTpy.functions import *
import gdspy
import copy


class EmptyGeometry(gdspy.PolygonSet, Validator, Settings):

    default_values = {}

    __slots__ = ('isInverted', 'name', '__position', '__angle', 'ports')

    def __init__(self, name='Empty Geometry', position=(0, 0), angle=0, *args, **kwargs):
        super().__init__([])
        self.isInverted = [False] * 256  # Массив с информацией о том, является ли слой инвертированным
        self.ports = []
        self.name = str(name)
        self.__position = (0, 0)  # начальное положение, относительно которого происходит отрисовка
        self.__angle = 0
        # апдейтим значания, получая часть от дефолтных, часть от введенных из kwargs
        values = copy.deepcopy(self.default_values)
        values.update(kwargs)
        # отрисовка
        self._drawing(values)
        # смещение и поворот
        self.__position = self.vector_2d(position)
        self.__angle = self.angle_quanted(angle)

    def _drawing(self, values):
        # Функция, содержащая рисовательную часть. Предполагается, что функция вызывается только в __init__.
        # Функцию следует строить так, чтобы отвязать ее от конкретных значений kwargs,
        # и в случае, когда ничего не передано, рисовать дефолтную схему
        pass

    # манипуляции с геометрией ______________________________

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

    def mirror(self, mirror_type='v'):
        if mirror_type not in ['v', 'h']:
            raise ValueError("mirror_type should be 'v' - vertical, or 'h' - horisontal")

        # переносим объект
        p1 = np.array(self.position)
        if mirror_type == 'v':
            p2 = p1 + np.array([0, 1])
            final_angle = np.angle(-np.cos(self.angle * np.pi / 180) + 1j * np.sin(self.angle * np.pi / 180), deg=True)
        else:
            p2 = p1 + np.array([1, 0])
            final_angle = np.angle(+np.cos(self.angle * np.pi / 180) - 1j * np.sin(self.angle * np.pi / 180), deg=True)

        # корректируем угол
        if final_angle < 0:
            final_angle += 360.
        self.__angle = final_angle

        for port in self.ports:
            delta_translation = np.array(port.position) - np.array(self.position)
            if mirror_type == 'v':
                final_angle = np.angle(-np.cos(port.angle * np.pi / 180) + 1j * np.sin(port.angle * np.pi / 180),
                                       deg=True)
                port.translate([-delta_translation[0]*2, 0])
            else:
                final_angle = np.angle(+np.cos(port.angle * np.pi / 180) - 1j * np.sin(port.angle * np.pi / 180),
                                       deg=True)
                port.translate([0, -delta_translation[1]*2])

            if final_angle < 0:
                final_angle += 360.
            port.angle = final_angle

        # изменяем полигоны
        super().mirror(p1, p2)

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

    def add_port(self, position, deg_angle):
        port = Port(self.vector_2d(position), self.angle_quanted(deg_angle))
        self.ports += [port]

    # гетеры/сетеры ________________________________________________

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

    def get_polygonset(self):
        result = gdspy.PolygonSet([])
        result.polygons = copy.deepcopy(self.polygons)
        result.layers = copy.deepcopy(self.layers)
        result.datatypes = copy.deepcopy(self.datatypes)
        return result

    #  отображение ___________________________________________

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"'{self.name}', " \
               f'({self.vector_2d(self.position)[0]}, {self.vector_2d(self.position)[1]}), ' \
               f'{self.angle} deg, ' \
               f'{len(self.ports)} ports\n'

    def geometry_string(self):
        return f'({self.vector_2d(self.position)[0]},\t{self.vector_2d(self.position)[1]}),\t' \
                          f'{self.angle_quanted(self.angle)} deg,\t' \
                          f'{len(self.ports)} ports\n'

    def show(self, ports=True, secondary_elements=True):
        # отрисовка основной геометрии
        gdspy.current_library = gdspy.GdsLibrary(name=self.name, unit=Settings.UNIT, precision=Settings.PRECISION)
        cell = gdspy.Cell(name=self.name)
        cell.add(self)

        # Отрисовка вспомогательных элементов
        bonding_box = self.get_bounding_box()
        if bonding_box is not None:
            width = bonding_box[1, 0] - bonding_box[0, 0]
            height = bonding_box[1, 1] - bonding_box[0, 1]
            # Просто характерный размер для вспомогательных элементов
            support_elements_size = round(min(width, height) / 5)
        else:
            support_elements_size = 1

        if secondary_elements:
            # центр объекта
            center_point = gdspy.Round(self.position, support_elements_size - support_elements_size / 10,
                                       initial_angle=np.pi,
                                       final_angle=3 * np.pi / 2,
                                       layer=Settings.PORT_LAYER)

            center_point = gdspy.boolean(center_point,
                                         gdspy.Round(self.position,
                                                     support_elements_size - support_elements_size / 10,
                                                     initial_angle=0,
                                                     final_angle=np.pi / 2), 'or', layer=Settings.PORT_LAYER)
            center_point.rotate(self.angle * np.pi / 180, self.position)
            cell.add(center_point)

            # центр координат
            center_point = gdspy.Round((0, 0), support_elements_size,
                                       inner_radius=support_elements_size - support_elements_size / 10,
                                       layer=Settings.PORT_LAYER)
            cell.add(center_point)
            center_point = gdspy.Round((0, 0), support_elements_size / 5,
                                       layer=Settings.PORT_LAYER)
            cell.add(center_point)

        # Отрисовка портов
        if ports:
            for i in range(0, len(self.ports)):
                label = gdspy.Label(str(i), self.ports[i].position, "nw", self.ports[i].angle,
                                    layer=Settings.PORT_LAYER)
                cell.add(label)

                a, o = self.ports[i].basis

                port_path = gdspy.Polygon([np.array(self.ports[i].position) + support_elements_size * np.array(a),
                                           np.array(self.ports[i].position) + support_elements_size * np.array(o) / 5,
                                           np.array(self.ports[i].position) - support_elements_size * np.array(o) / 5],
                                          layer=Settings.PORT_LAYER)
                cell.add(port_path)

        gdspy.LayoutViewer(library=None, cells=cell)
        del cell

    #  сохранение ___________________________________________

    def save_gds(self, filename):
        gdspy.current_library = gdspy.GdsLibrary(name=self.name, unit=Settings.UNIT, precision=Settings.PRECISION)
        cell = gdspy.Cell(name=self.name)
        cell.add(self)
        gdspy.current_library.write_gds(filename + '.gds')

    def save_svg(self, filename):
        cell = gdspy.Cell(name=self.name)
        cell.add(self)
        cell.write_svg(filename + '.svg')

    #  булевы операции (см. functions.py)_____________________

    def boolean(self, tool_obj, operation):
        return boolean(self, tool_obj, operation, keep_obj=False,
                       precision=Settings.PRECISION, max_points=Settings.MAX_POLYGON_POINTS)

    def __mul__(self, tool_obj):
        return self.boolean(tool_obj, 'and')

    def __sub__(self, tool_obj):
        return self.boolean(tool_obj, 'not')

    def __add__(self, tool_obj):
        return self.boolean(tool_obj, 'or')

    def __truediv__(self, tool_obj):
        return self.boolean(tool_obj, 'xor')

    def append(self, tool_obj):
        append_geometry(self, tool_obj)


