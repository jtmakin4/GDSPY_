from EDTpy.geometry import *
from EDTpy.functions import *
from EDTpy.settings import *

import gdspy
import copy
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_nb
from collections.abc import Iterable


# todo сделать возможным болеан скетчей
class EmptySketch(EmptyGeometry):
    __slots__ = ('__geometries',)

    def __init__(self, name='Empty Sketch', position=(0, 0), angle=0, *args, **kwargs):
        super().__init__(name=name, position=position, angle=angle, *args, **kwargs)
        self.__geometries = {}

    @property
    def geometries(self):
        return self.__geometries

    def clear_geometries(self):
        self.__geometries = {}

    def add_geometry(self, geometry):
        if not isinstance(geometry, Iterable):
            if geometry.name in self.__geometries.keys():
                self.__geometries[geometry.name] += [geometry]
            else:
                self.__geometries[geometry.name] = [geometry]
        else:
            for geom in geometry:
                if geom.name in self.__geometries.keys():
                    self.__geometries[geom.name] += [geom]
                else:
                    self.__geometries[geom.name] = [geom]

    def geometry_list(self):
        result = []
        for key in self.geometries.keys():
            result += self.geometries[key]
        return result

    def assemble(self, keep_polygones=False, keep_unused_layers=False, keep_unused_datatypes=False):
        # todo сделать так, чтобы запись шла только по полигонам, присутствующим в Sketch
        layers = unique_layers(self.geometry_list())
        datatypes = unique_datatypes(self.geometry_list())

        # задаем progtess bar
        try:
            pbar = tqdm(self.geometry_list(), desc=f"Assembling {self.name}")
        except AttributeError:
            pbar = tqdm_nb(self.geometry_list(), desc=f"Assembling {self.name}")

        for geom in self.geometry_list():
            for layer in layers:
                for datatype in datatypes:
                    if not geom.isInverted[layer]:
                        if keep_polygones:
                            self.append(extract(geom, layer, datatype, keep_obj=True))
                        else:
                            self + extract(geom, layer, datatype, keep_obj=True)
                    else:
                        self - extract(geom, layer, datatype, keep_obj=True)
            pbar.update(1)

    def linear_array(self, geometries, vec1, n1, vec2=(0, 0), n2=1):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        if not isinstance(geometries, Iterable):
            geometries = [geometries]
        else:
            geometries = list(geometries)

        for geom in geometries:
            for i in range(0, n1):
                for j in range(0, n2):
                    if i != 0 or j != 0:
                        obj = copy.deepcopy(geom)
                        obj.translate(dposition=vec1*i + vec2*j)
                        self.add_geometry(obj)

    def circular_array(self, geometries, center=(0, 0), num=4):
        if not isinstance(geometries, Iterable):
            geometries = [geometries]
        else:
            geometries = list(geometries)

        for geom in geometries:
            for i in range(1, num):
                obj = copy.deepcopy(geom)
                obj.rotate(angle=360./num*i, center=center)
                self.add_geometry(obj)

    def mirror_geometries(self, geometries=None, mirror_type='h'):
        if mirror_type not in ['v', 'h']:
            raise ValueError("mirror_type should be 'v' - vertical, or 'h' - horisontal")

        if geometries is None:
            geometries = self.geometry_list()

        if not isinstance(geometries, Iterable):
            geometries = [geometries]
        else:
            geometries = list(geometries)

        for geom in geometries:
            obj = copy.deepcopy(geom)
            obj.mirror(mirror_type)
            if mirror_type == 'v':
                obj.position = (-obj.position[0], obj.position[1])
            else:
                obj.position = (obj.position[0], -obj.position[1])
            self.add_geometry(obj)

    def show(self, ports=True, secondary_elements=False):
        # отрисовка основной геометрии
        gdspy.current_library = gdspy.GdsLibrary(name=self.name, unit=Settings.UNIT, precision=Settings.PRECISION)
        show_cell = gdspy.Cell(name=self.name)
        show_cell.add(self)

        # Отрисовка вспомогательных элементов
        bonding_box = self.get_bounding_box()
        if bonding_box is not None:
            width = bonding_box[1, 0] - bonding_box[0, 0]
            height = bonding_box[1, 1] - bonding_box[0, 1]
            support_elements_size = round(
                min(width, height) / 10)  # Просто характерный размер для вспомогательных элементов
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
            show_cell.add(center_point)

            # центр координат
            center_point = gdspy.Round((0, 0), support_elements_size,
                                       inner_radius=support_elements_size - support_elements_size / 10,
                                       layer=Settings.PORT_LAYER)
            show_cell.add(center_point)
            center_point = gdspy.Round((0, 0), support_elements_size / 5,
                                       layer=Settings.PORT_LAYER)
            show_cell.add(center_point)

        # Отрисовка портов
        if ports:
            for i in range(0, len(self.ports)):
                label = gdspy.Label(str(i), self.ports[i].position, "nw", self.ports[i].angle,
                                    layer=Settings.PORT_LAYER)
                show_cell.add(label)

                a, o = self.ports[i].basis

                port_path = gdspy.Polygon([np.array(self.ports[i].position) + support_elements_size * np.array(a) / 2,
                                           np.array(self.ports[i].position) + support_elements_size * np.array(o) / 10,
                                           np.array(self.ports[i].position) - support_elements_size * np.array(o) / 10],
                                          layer=Settings.PORT_LAYER)
                show_cell.add(port_path)

            for geom in self.geometry_list():
                for i in range(0, len(geom.ports)):
                    label = gdspy.Label(str(i), geom.ports[i].position, "nw", geom.ports[i].angle,
                                        layer=Settings.PORT_LAYER)
                    show_cell.add(label)

                    a, o = geom.ports[i].basis

                    port_path = gdspy.Polygon([np.array(geom.ports[i].position) + support_elements_size * np.array(a)/2,
                                               np.array(geom.ports[i].position) + support_elements_size * np.array(
                                                   o) / 10,
                                               np.array(geom.ports[i].position) - support_elements_size * np.array(
                                                   o) / 10],
                                              layer=Settings.PORT_LAYER)
                    show_cell.add(port_path)

        gdspy.LayoutViewer(library=None, cells=show_cell)
        del show_cell
        return self.__str__()

    def get_content(self, full=True):
        divider_len = 50
        print('_' * divider_len)
        print(f"'{self.name}'\n{len(self.geometry_list())} geometries:")

        for key in self.geometries.keys():
            print(f"\t - '{key}': {len(self.geometries[key])}")
        print(f'{len(self.ports)} ports')
        print('.' * divider_len)
        if full:
            if len(self.geometries.keys()) != 0:
                for key in self.geometries.keys():
                    print(f"'{key}':")
                    for i, geom in enumerate(self.geometries[key]):
                        print(f'\t{i}:\t{geom.geometry_string()}', end='')
                    print('.' * divider_len)
            else:
                print('No geometries')
                print('.' * divider_len)

            if len(self.ports) != 0:
                print('Sketch ports:')
                for i, port in enumerate(self.ports):
                    port_string = f'({port.vector_2d(port.position)[0]}, {port.vector_2d(port.position)[1]}), ' \
                               f'{port.angle} deg'
                    print(f'\t{i}:\t{port_string}', end='')
            else:
                print('No sketch ports')
            print('_' * divider_len)
