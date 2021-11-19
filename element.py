import gdspy
import numpy as np

curve_tolerance = 5e0
curve_precision = 5e-1
port_layer = 4

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

        # basis vectors


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

    def basis(self):
        a = np.array([np.cos(self.angle), np.sin(self.angle)])
        o = np.array([np.cos(self.angle+np.pi), np.sin(self.angle+np.pi)])
        return a, o

    def position(self):
        return np.array(self.position)


class Geometry:
    def __init__(self, name='Empty Geometry', unit=1e-6, precision=1e-9):
        self.elements = [gdspy.PolygonSet([], layer=i) for i in range(0, 255)]
        self.layers = [i for i in range(0, 255)]
        self.isInverted = [False for i in range(0, 255)] # Массив с информацией о том, является ли слой инвертированным
        self.portsLayer = port_layer
        self.ports = []

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

    def rotate(self, dangle):
        if self.float:
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
            port_path = gdspy.RobustPath((0, 0), 0, layer=self.portsLayer)
            a, _ = self.ports[i].basis()
            print(self.ports[i].position)
            port_path.parametric(lambda u: np.array(self.ports[i].position) + a*u*3e3, width=lambda u: 1e3*(1-u))
            cell.add(port_path)

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
        if self.float:
            self.ports.append(Port(position, angle))

    def mergeWithPort(self, aim_port, port_num):
        if self.float:
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
        if self.float:
        #todo сделать проверку на то, что boolObj это PlanarGeometry
            self.elements[layer] = gdspy.boolean(self.elements[layer], boolObj, operand, layer=layer)


class CPW(Geometry):
    # копланары должны создаваться уже внутри скетча,  но в принципе можно задавать как хочешь
    def __init__(self, port1, port2, s, w, r=None, gap=300, inter_dist=300, via_d=300):
        super().__init__(name='cpw')
        self.isInverted[0] = True
        self.isInverted[2] = True
        self.port1 = port1
        self.port2 = port2

        self._s = s
        self._w = w
        self._g = gap
        self.inter_dist = inter_dist
        self.via_d = via_d


        self._path, self.Rmax = self.getTrajectory()

        if r != None:
            if r > self.Rmax:
                raise ValueError
            else:
                self._r = r
        else:
            self._r = self.Rmax

        self._in_point = self._path[0]
        self._out_point = self._path[-1]

        w_polygon = gdspy.FlexPath(self._path,
                                   self._s + 2 * self._w,
                                   corners="circular bend",
                                   bend_radius=self._r,
                                   tolerance=curve_tolerance,
                                   precision=curve_precision)

        s_polygon = gdspy.FlexPath(self._path,
                                   self._s,
                                   corners="circular bend",
                                   bend_radius=self._r,
                                   tolerance=curve_tolerance,
                                   precision=curve_precision)

        tech_polygon = gdspy.FlexPath(self._path,
                               self._s + 2 * self._w + 2 * (300 + self.via_d),
                               corners="circular bend",
                               bend_radius=self._r,
                               tolerance=curve_tolerance,
                               precision=curve_precision)

        w_polygon = gdspy.boolean(w_polygon, s_polygon, 'not')

        vias_polygon = gdspy.PolygonSet([])

        offset = self.via_d + 300 + self._s / 2 + self._w

        via_centers = self._create_via_centers(self._path, self._r, offset, self.inter_dist)
        for i in range(0, len(via_centers)):
            arc = gdspy.Round(
                via_centers[i],
                via_d,
                tolerance=curve_tolerance * 0.1,
            )
            vias_polygon = gdspy.boolean(vias_polygon, arc, 'or')


        # Рисуем
        self.boolean(vias_polygon, 'or', 3)
        self.boolean(w_polygon, 'or', 0)
        self.boolean(tech_polygon, 'or', 5)

    def getTrajectory(self):
        # строит траекторию между двумя портами, соединяя их двумя дугами окружностей, касающихся друг друга и имеющих наибольши радиус
        # todo сделать проверку на то, стоит ли добавлять дополнительные скругления если порты смотрят в разные стороны

        # todo сделать проверку на нахождение на одной прямой
        Rmax = 0
        s1 = None
        s2 = None
        sign1 = [-1, 1]
        sign2 = [-1, 1]

        a1, o1 = self.port1.basis()
        r1 = self.port1.position()
        a2, o2 = self.port2.basis()
        r2 = self.port2.position()
        delt_r = r1 - r2

        # находим радиус и положение центров окружностей
        for i in sign1:
            for j in sign2:
                if (1 + i * j * np.dot(o1, o2)) > 1e-9:
                    delt_o = i * o1 - j * o2
                    D = np.dot(delt_o, delt_r)**2
                    D += 2 * np.dot(delt_r, delt_r) * (1 + i*j*np.dot(o1, o2))
                    R = np.dot(delt_o, delt_r) + np.sqrt(D)
                    R /= (2*(1 + i*j*np.dot(o1, o2)))
                else:
                    R = -np.dot(delt_r, delt_r)/2/np.dot(delt_r, delt_r)

                if R >= Rmax and R > 0:
                    Rmax = R
                    s1 = i
                    s2 = j

        c1 = r1 + Rmax * s1 * o1
        c2 = r2 + Rmax * s2 * o2
        delt_c = c1-c2
        p = (c1 + c2)/2

        # Создаем промужуточные точки на кривой
        x1 = np.dot(p-r1, delt_c)/np.dot(a1, delt_c)
        x2 = np.dot(p - r2, delt_c) / np.dot(a2, delt_c)
        path = [r1, r1 + a1*x1, r2 + a2*x2, r2]

        return path, Rmax

    # Старые функции для нахождения положения сквозных отверстий
    def _circular_bend_path(self, input_array, bend_r):

        def _unit_vector(point1, point2):
            vec = np.array(point2) - np.array(point1)
            vec = vec / np.linalg.norm(vec)
            return vec

        def _array_simplify(input_array):
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

    def _point_to_len_array(self, points, angles, bend_r):
        length_array = np.array([0])
        for i in range(1, len(points)):
            if i % 2 == 1:
                vec = np.array(points[i] - points[i - 1])
                length_array = np.append(length_array, np.linalg.norm(vec) + length_array[-1])
            else:
                length_array = np.append(length_array, bend_r * angles[i // 2 - 1] + length_array[-1])
        return length_array[1:]

    def _len_to_position(self, l, points, centers, angles, bend_r):

        def _rotate_around(point, center, angle, diraction):
            if diraction > 0:
                rotate_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                          [np.sin(angle), np.cos(angle)]])
            else:
                rotate_matrix = np.array([[np.cos(angle), np.sin(angle)],
                                          [-np.sin(angle), np.cos(angle)]])
            return np.dot(rotate_matrix, point - center) + center

        length_array = self._point_to_len_array(points, angles, bend_r)
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
            return result_point, vec

    def _create_via_centers(self, path, bend_radius, offset, step):
        path_array, bend_centers, bend_angles = self._circular_bend_path(path, bend_radius)
        length_arr = self._point_to_len_array(path_array, bend_angles, bend_radius)
        l_arr = np.arange(0, length_arr[-1], step)
        points_array = np.array([0, 0])
        for l in l_arr:
            p, vec = self._len_to_position(l, path_array, bend_centers, bend_angles, bend_radius)
            points_array = np.vstack([points_array, p + vec * offset])
            points_array = np.vstack([points_array, p - vec * offset])
        # todo check condition
        return points_array[1:]

class Mounting(Geometry):
    # по идее в подобных дочерних классах должна быть только их инициализация, остольные функции прописаны в классе-родителе. Возможно это можно сделать еще более просто.
    def __init__(self):
        super().__init__()

        # Рисование происходит здесь



        # записываем результат

        # задаем порты
        self.addPort(position=(0, 0), angle=0.)
