from EDTpy.geometry import *
from EDTpy.validator import *
from EDTpy.port import *
import gdspy
import copy


class EmptyPath(EmptyGeometry, Validator):

    __slots__ = ('__path', '__rounded_path', '__len_array', '__spots',
                 '__radiuses', '__angles', '__centers', '__length')

    def __init__(self, path, radius, *args, **kwargs):
        # проверка и упрощние заданной ломаной
        self.__path = self.simplify_array(path)
        # print((len(self.__path)))  # todo проблема с прямыми линиями
        # проверка и преобразование радиусов
        if not isinstance(radius, Iterable):
            # здесь я оставил возможность задавать различные радиусы, но это не отрисовать в FlexPath,
            # поэтому возможность временно отрублена. Алгоритмы принимают именно массив из скруглений
            self.__radiuses = [radius]*(len(self.__path)-2)
        else:
            raise ValueError('radius should be integer or float')

        # задание скруглений
        self.__rounded_path, self.__centers, self.__angles = self._circular_bend_path(self.__path, self.__radiuses)
        # __rounded_path - Это ломаная, у которой вырезаны углы по точкам касания

        # определение длинны линии
        self.__len_array = self._get_len_array(self.__rounded_path, self.__radiuses, self.__angles)
        self.__length = self.__len_array[-1]

        # задание спотов
        # Споты - порты для размещения элементов вроде мостов, via holes, прикрепленные к кривой
        # Споты задаются при отрисовке (функция _drawing)
        self.__spots = []

        # Важно! отрисовка геометрии происходит только в конце, после задания кривой, так как иначе элементы,
        # привязанные к кривой, не смогут построиться

        super().__init__(name='Path', position=(0, 0), angle=0, *args, **kwargs)

    @property
    def rounded_path(self):
        return self.__rounded_path

    @property
    def centers(self):
        return self.__centers

    @property
    def angles(self):
        return self.__angles

    @property
    def len_array(self):
        return self.__len_array

    @property
    def length(self):
        return self.__length

    @property
    def spots(self):
        return self.__spots

    @property
    def path(self):
        return self.__path

    @property
    def radiuses(self):
        return self.__radiuses

    def set_spots(self, length, relative=False, additional_angle=0, on_line_segment=True, on_curve_segment=True):
        # для нахождение положения на кривой используется вызов self() (см. функция __call__)
        positions = self(length, relative=False)
        for point in positions:
            coordinates, norm_vec, on_line = point
            on_curve = not on_line
            angle = np.angle(norm_vec[0] + 1j*norm_vec[1])*180/np.pi
            if angle < 0:
                angle += 360
            angle += additional_angle
            if (on_line_segment and on_line) or (on_curve_segment and on_curve):
                self.__spots += [Port(coordinates, angle)]

    def show(self, ports=True, spots=False):
        # отрисовка основной геометрии
        gdspy.current_library = gdspy.GdsLibrary(name=self.name, unit=Settings.UNIT, precision=Settings.PRECISION)
        cell = gdspy.Cell(name=self.name)
        cell.add(self)

        # Отрисовка вспомогательных элементов
        bonding_box = self.get_bounding_box()
        if bonding_box is not None:
            width = bonding_box[1, 0] - bonding_box[0, 0]
            height = bonding_box[1, 1] - bonding_box[0, 1]
            support_elements_size = round(
                min(width, height) / 5)  # Просто характерный размер для вспомогательных элементов
        else:
            support_elements_size = 1

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

        # Отрисовка портов
        if spots:
            for i in range(0, len(self.spots)):
                label = gdspy.Label(str(i), self.spots[i].position, "nw", self.spots[i].angle,
                                    layer=Settings.PORT_LAYER)
                cell.add(label)

                a, o = self.spots[i].basis

                port_path = gdspy.Polygon([np.array(self.spots[i].position) + support_elements_size * np.array(a),
                                           np.array(self.spots[i].position) + support_elements_size * np.array(o)/15,
                                           np.array(self.spots[i].position) - support_elements_size * np.array(o)/15],
                                          layer=Settings.PORT_LAYER)
                cell.add(port_path)

        gdspy.LayoutViewer(library=None, cells=cell)
        del cell
        return self.__str__()

    # ______________ функции для нахождения отображение len position -> coordinates _______________
    @classmethod
    def unit_vector(cls, point1, point2):
        vec = np.array(point2) - np.array(point1)
        vec = vec / np.linalg.norm(vec)
        return vec

    @classmethod
    def simplify_array(cls, input_array):
        # удаляет случайные лишние точки прямой
        array = np.array(input_array[0])
        for i in range(1, len(input_array) - 1):
            vec1 = cls.unit_vector(input_array[i - 1], input_array[i])
            vec2 = cls.unit_vector(input_array[i], input_array[i + 1])
            if vec1.dot(vec2) != 1:
                array = np.vstack([array, input_array[i]])
        array = np.vstack([array, input_array[-1]])
        return array

    @classmethod
    def _circular_bend_path(cls, input_array, bend_r):
        input_array = cls.simplify_array(input_array)
        if not isinstance(bend_r, Iterable):
            bend_r = [bend_r]*(len(input_array)-2)
        elif len(bend_r) + 2 != len(input_array):
            raise IndexError('radiuses number is not equal to the number of path vertexes')
        else:
            pass

        array = np.array(input_array[0])
        arc_centers = np.array([0, 0])
        angles = np.array([])

        # todo переделать так, чтобы можно было впилить проверку на радиус
        for i in range(1, len(input_array) - 1):
            vec1 = cls.unit_vector(input_array[i - 1], input_array[i])
            vec2 = cls.unit_vector(input_array[i], input_array[i + 1])
            cos = vec1.dot(vec2)
            if cos != 1:
                alpha = np.arccos(cos)
                a = bend_r[i - 1] / np.tan(np.pi / 2 - alpha / 2)
                new_point1 = -vec1 * a + input_array[i]
                new_point2 = vec2 * a + input_array[i]
                array = np.vstack([array, new_point1, new_point2])

                angles = np.append(angles, alpha)
                if np.cross(vec1, vec2) < 0:
                    arc_centers = np.vstack([arc_centers, np.array([vec1[1], -vec1[0]]) * bend_r[i - 1] + new_point1])
                else:
                    arc_centers = np.vstack([arc_centers, np.array([-vec1[1], vec1[0]]) * bend_r[i - 1] + new_point1])
            else:
                array = np.vstack([array, input_array[i]])
        array = np.vstack([array, input_array[-1]])

        # todo да, костыльно задавать так arc_centers. Но я пока не разобрался как правильно это делать
        return array, arc_centers[1:], angles

    @classmethod
    def _get_len_array(cls, points, bend_r, angles):

        # проверка и преобразование радиусов
        if not isinstance(bend_r, Iterable):
            bend_r = [bend_r]*int((len(points)/2-1))
        elif len(bend_r) != len(points)/2-1:
            raise IndexError('bend_r length is not equal with number of path vertexes')
        else:
            pass

        length_array = np.array([0])
        for i in range(1, len(points)):
            if i % 2 == 1:
                vec = np.array(points[i] - points[i - 1])
                length_array = np.append(length_array, np.linalg.norm(vec) + length_array[-1])
            else:
                length_array = np.append(length_array, bend_r[i//2-1] * angles[i // 2 - 1] + length_array[-1])
        return length_array[1:]

    def len_position(self, length, relative=False):
        points = self.rounded_path
        centers = self.centers
        angles = self.angles
        length_array = self.len_array
        section = 0
        if relative:
            if length > 1 or length < 0:
                raise ValueError('in relative regime length should be in [0,1]')
            else:
                length = self.length*length

        def _rotate_around(point, center, angle, diraction):
            if diraction > 0:
                rotate_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                          [np.sin(angle), np.cos(angle)]])
            else:
                rotate_matrix = np.array([[np.cos(angle), np.sin(angle)],
                                          [-np.sin(angle), np.cos(angle)]])
            return np.dot(rotate_matrix, point - center) + center

        if length > self.length:
            print("asked length is more than overall length")
            return None

        # определяем, к какому участку относится значение
        if length < length_array[0]:
            section = 0

        for i in range(1, len(length_array)):
            if length_array[i - 1] < length <= length_array[i]:
                section = i

        if section % 2 == 0:
            on_line_segment = True
            vec = points[section + 1] - points[section]
            vec = vec / np.linalg.norm(vec)
            result_point = points[section + 1] + vec * (length - length_array[section])
            return result_point, vec, on_line_segment
        else:
            on_line_segment = False
            vec1 = points[section] - points[section - 1]
            vec2 = points[section + 2] - points[section + 1]
            diraction = np.cross(vec1, vec2)
            point_angle = (length - length_array[section - 1]) /\
                          (length_array[section] - length_array[section - 1]) * angles[section // 2]
            result_point = _rotate_around(points[section], centers[section // 2], point_angle, diraction)
            vec = np.dot(np.array([[0, -1], [1, 0]]), np.array(result_point - centers[section // 2]) / np.linalg.norm(
                result_point - centers[section // 2]))
            if diraction > 0:
                return result_point, vec, on_line_segment
            else:
                return result_point, -vec, on_line_segment

    def __call__(self, length, relative=False):
        # экземпляр класса можно вызывать как функцию, которая является векторизованной len_position
        # и позволяет используется для построения спотов
        if not isinstance(length, Iterable):
            length = [length]
        result = [self.len_position(i, relative) for i in length]
        return result

    def geometry_string(self):
        string = f'{self.coordinate(self.length)},\t'
        for point in self.__path:
            string += f'({point[0]},\t{point[1]}),\t'
        string += f'{self.value_quanted(self.radiuses[0])}\n'
        return string

    def mirror(self, mirror_type='v'):
        super().mirror(mirror_type)
        for point in self.__path:
            if mirror_type == 'v':
                point[0] = -point[0]
            else:
                point[1] = -point[1]

    # ______________________________________________________________________________________________
