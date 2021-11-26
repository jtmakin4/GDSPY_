import gdspy
import numpy as np

curve_tolerance = 1e0
curve_precision = 1e-1
port_layer = 4


class Port:
    """
    Объект Порт, являющийся дополнениям к PolygonSet пакета Numpy
    """
    def __init__(self, position, angle):
        if len(position) == 2:
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
        if len(dposition) == 2:
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
        o = np.array([np.cos(self.angle+np.pi/2), np.sin(self.angle+np.pi/2)])
        return a, o


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

    def rotate(self, dangle, center=None):
        if center == None:
            if self.float:
                self.elements = [element.rotate(dangle, center=self.position) for element in self.elements]
                for i in range(0, len(self.ports)):
                    self.ports[i].rotate(dangle, center=self.position)
                self.angle += dangle
        else:
            if self.float:
                self.elements = [element.rotate(dangle, center=center) for element in self.elements]
                for i in range(0, len(self.ports)):
                    self.ports[i].rotate(dangle, center=center)
                self.angle += dangle

    def draw(self):
        gdspy.current_library = gdspy.GdsLibrary(name='', unit=self.unit, precision=self.precision)
        cell = gdspy.Cell(name=self.name)
        cell.add(self.elements)
        for i in range(0, len(self.ports)):
            label = gdspy.Label(str(i), self.ports[i].position, "nw", self.ports[i].angle, layer=self.portsLayer)
            cell.add(label)
            # port_path = gdspy.RobustPath((0, 0), 0, layer=self.portsLayer)
            a, o = self.ports[i].basis()
            # print(self.ports[i].position)
            # port_path.parametric(lambda u: np.array(self.ports[i].position) + a*u*3e3, width=lambda u: 1e3*(1-u))
            port_path = gdspy.Polygon([np.array(self.ports[i].position) + 1e3*a,
                                       np.array(self.ports[i].position) + 0.2e3*o,
                                       np.array(self.ports[i].position) - 0.2e3*o], layer=self.portsLayer)
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
                                                 layer=layer, precision=curve_precision)
        # сглаживание
        self.elements[layer] = gdspy.boolean(self.elements[layer].polygons, None, 'or', max_points=0,
                                             layer=layer, precision=curve_precision)


class CPW(Geometry):
    # копланары должны создаваться уже внутри скетча,  но в принципе можно задавать как хочешь
    def __init__(self, port1, port2, s, w, r=None, gap=300, inter_dist=300, via_r=300/2, path=None, chessOrder = False):
        # Оставил возможность задавать свою траекторию
        super().__init__(name='cpw')
        self.isInverted[0] = True
        self.isInverted[2] = True
        self.port1 = port1
        self.port2 = port2

        self._s = s
        self._w = w
        self._g = gap
        # self.inter_dist = inter_dist + via_r*2
        self.via_r = via_r

        if path is None:
            self._path, self.Rmax = self.getTrajectory()

            if r is not None:
                if r > self.Rmax:
                    raise ValueError
                else:
                    self._r = r
            else:
                self._r = self.Rmax
        else:
            self._path = path
            if r is not None:
                self._r = r
            else:
                raise ValueError

        self._in_point = self._path[0]
        self._out_point = self._path[-1]

        w_polygon = gdspy.FlexPath(self._path,
                                   self._s + 2 * self._w,
                                   corners="circular bend",
                                   bend_radius=self._r,
                                   tolerance=curve_tolerance,
                                   precision=curve_precision)

        sentral_line = gdspy.FlexPath(self._path,
                                   0.01,
                                   corners="circular bend",
                                   bend_radius=self._r,
                                   tolerance=curve_tolerance,
                                   precision=curve_precision)

        self.length = round(sentral_line.area()/0.01, 0)
        viaNum = self.length/(inter_dist+2*via_r)
        viaNum = int(round(viaNum, 0))
        self.inter_dist = self.length/viaNum

        s_polygon = gdspy.FlexPath(self._path,
                                   self._s,
                                   corners="circular bend",
                                   bend_radius=self._r,
                                   tolerance=curve_tolerance,
                                   precision=curve_precision)

        # tech_polygon = gdspy.FlexPath(self._path,
        #                               self._s + 2 * self._w + 2 * 300,
        #                               corners="circular bend",
        #                               bend_radius=self._r,
        #                               tolerance=curve_tolerance,
        #                               precision=curve_precision)

        # tech_polygon = gdspy.FlexPath(self._path,
        #                               300,
        #                               offset=[self._s/2 + self._w + 300 + 300/2, -self._s/2 - self._w - 300 - 300/2],
        #                               corners="circular bend",
        #                               ends='round',
        #                               bend_radius=self._r,
        #                               tolerance=curve_tolerance,
        #                               precision=curve_precision)

        w_polygon = gdspy.boolean(w_polygon, s_polygon, 'not')

        via_polygon = gdspy.PolygonSet([])

        offset = self.via_r + 300 + self._s / 2 + self._w

        via_centers_left, via_centers_right = self._create_via_centers(self._path, self._r, offset, self.inter_dist)

        if not chessOrder:
            for i in range(0, len(via_centers_left)):
                arc = gdspy.Round(
                    via_centers_left[i],
                    via_r,
                    tolerance=curve_tolerance * 0.1,
                )
                via_polygon = gdspy.boolean(via_polygon, arc, 'or')
            for i in range(0, len(via_centers_right)):
                arc = gdspy.Round(
                    via_centers_right[i],
                    via_r,
                    tolerance=curve_tolerance * 0.1,
                )
                via_polygon = gdspy.boolean(via_polygon, arc, 'or')
        else:
            for i in range(0, len(via_centers_left)):
                if i % 2 == 0:
                    arc = gdspy.Round(
                        via_centers_left[i],
                        via_r,
                        tolerance=curve_tolerance * 0.1,
                    )
                    via_polygon = gdspy.boolean(via_polygon, arc, 'or')
            for i in range(0, len(via_centers_right)):
                if i % 2 == 1:
                    arc = gdspy.Round(
                        via_centers_right[i],
                        via_r,
                        tolerance=curve_tolerance * 0.1,
                    )
                    via_polygon = gdspy.boolean(via_polygon, arc, 'or')


        # Рисуем
        self.boolean(via_polygon, 'or', 3)
        self.boolean(w_polygon, 'or', 0)
        # self.boolean(tech_polygon, 'or', 5)

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
        r1 = np.array(self.port1.position)
        a2, o2 = self.port2.basis()
        r2 = np.array(self.port2.position)
        delt_r = r1 - r2

        if np.dot(-delt_r, o1) > 0:
            s1 = 1
        elif np.dot(-delt_r, o1) < 0:
            s1 = -1
        else:
            s1 = 0
        # todo добавить обработку случая, s1 = 0, s2 != 0
        if np.dot(delt_r, o2) > 0:
            s2 = 1
        elif np.dot(delt_r, o2) < 0:
            s2 = -1
        else:
            s2 = 0

        # s1 = -1
        # s2 = -1

        if s1 == 0 and s2 == 0:
            path = [r1,  r2]
            Rmax = None
        elif (s1 != 0 and s2 == 0) or (s1 == 0 and s2 != 0):
            pass
            # по идее здесь будет кривая с произвольным скруглением, хз
        else:
            if (1 + s1 * s2 * np.dot(o1, o2)) > 1e-9:
                delt_o = s1 * o1 - s2 * o2
                D = np.dot(delt_o, delt_r) ** 2
                D += 2 * np.dot(delt_r, delt_r) * (1 + s1 * s2 * np.dot(o1, o2))
                Rmax = np.dot(delt_o, delt_r) + np.sqrt(D)
                Rmax /= (2 * (1 + s1 * s2 * np.dot(o1, o2)))
            else:
                Rmax = -np.dot(delt_r, delt_r) / 2 / np.dot(delt_r, delt_r)

            Rmax = round(Rmax/1000, 1)*1000
            # print(Rmax)
            c1 = r1 + Rmax * s1 * o1
            c2 = r2 + Rmax * s2 * o2
            delt_c = c1 - c2
            p = (c1 + c2) / 2
            # print(s1, s2)
            # Создаем промужуточные точки на кривой
            x1 = np.dot(p - r1, delt_c) / np.dot(a1, delt_c)
            x2 = np.dot(p - r2, delt_c) / np.dot(a2, delt_c)
            path = [r1, r1 + a1 * x1, r2 + a2 * x2, r2]

        return path, Rmax*0.99

    # Старые функции для нахождения положения сквозных отверстий
    # todo здесь не обработан случай, когда кривая состтоит только из окружностей возможно внутри уменьшать радиус на 0.01, что не повлияет на геометрию, но все равно норм будет
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
            if diraction > 0:
                return result_point, vec
            else:
                return result_point, -vec

    def _create_via_centers(self, path, bend_radius, offset, step):
        path_array, bend_centers, bend_angles = self._circular_bend_path(path, bend_radius)
        length_arr = self._point_to_len_array(path_array, bend_angles, bend_radius)
        l_arr = np.arange(0, length_arr[-1], step)
        points_array_left = np.array([0, 0])
        points_array_right = np.array([0, 0])
        for l in l_arr:
            p, vec = self._len_to_position(l, path_array, bend_centers, bend_angles, bend_radius)
            points_array_left = np.vstack([points_array_left, p + vec * offset])
            points_array_right = np.vstack([points_array_right, p - vec * offset])
        # todo check condition
        return points_array_left[1:], points_array_right[1:]


class SMP_PCB(Geometry):
    # по идее в подобных дочерних классах должна быть только их инициализация, остольные функции прописаны в классе-родителе. Возможно это можно сделать еще более просто.
    def __init__(self):
        super().__init__()
        self.name = 'mounting place for SMP-PCB connector'

        # Рисование происходит здесь
        for i in [0, 1, 2]:
            self.isInverted[i] = True

        # сквозные отверстия
        nViaT = 18
        rViaT = 300 / 2
        rViaPosT = 2000
        viaGapLimit = 300
        s = 1000
        w = 330
        gap = 300

        startAngle = np.arcsin((rViaT + +s / 2 + w + gap) / rViaPosT)
        singleViaAng = (np.pi * 2 - 2 * startAngle) / (nViaT - 1)

        singleVia = gdspy.Round((rViaPosT * np.cos(startAngle), rViaPosT * np.sin(startAngle)), rViaT,
                                tolerance=curve_tolerance)

        for i in range(0, nViaT):
            self.boolean(singleVia, 'or', 3)
            singleVia.rotate(singleViaAng)

        rs = 1050 / 2
        rw = rs + 600

        distVia = 300
        nViaLen = 3
        length = rViaPosT * np.cos(startAngle) + (distVia + 2 * rViaT) * nViaLen

        singleViaDown = gdspy.Round((rViaPosT * np.cos(startAngle), -rViaPosT * np.sin(startAngle)), rViaT,
                                    tolerance=curve_tolerance)
        singleViaUp = gdspy.Round((rViaPosT * np.cos(startAngle), rViaPosT * np.sin(startAngle)), rViaT,
                                  tolerance=curve_tolerance)
        for i in range(0, nViaLen):
            singleViaDown.translate(distVia + 2 * rViaT, 0)
            self.boolean(singleViaDown, 'or', 3)
            singleViaUp.translate(distVia + 2 * rViaT, 0)
            self.boolean(singleViaUp, 'or', 3)

        # разъем со стороны копланаров
        cpw = gdspy.Round((0, 0), rw)
        cpw = gdspy.boolean(cpw, gdspy.Rectangle((0, s / 2 + w), (length, -s / 2 - w)), 'or')
        cpw = gdspy.boolean(cpw, gdspy.Round((0, 0), rs, tolerance=curve_tolerance), 'not')
        cpw = gdspy.boolean(cpw, gdspy.Rectangle((0, s / 2), (length, -s / 2)), 'not')

        self.boolean(cpw, 'or', 0)

        # разъем со стороны коннектора
        rsb = 1100 / 2
        rwb = rs + 930
        self.boolean(gdspy.Round((0, 0), rwb, inner_radius=rsb, tolerance=curve_tolerance), 'or', 2)

        # центральное отверстие
        rhole = 800 / 2
        for i in [0, 1, 2]:
            self.boolean(gdspy.Round((0, 0), rhole), 'or', i)

        # крепления
        rHMount = 1000 / 2
        aHMount = 5080
        rTHInner = rHMount + 200
        rTHOuter = rTHInner + 200

        for i in [-1, 1]:
            for j in [-1, 1]:
                for k in [0, 1, 2]:
                    self.boolean(gdspy.Round((i * aHMount / 2, j * aHMount / 2), rHMount, tolerance=curve_tolerance),
                                 'or', k)

        self.boolean(gdspy.Round((aHMount / 2, aHMount / 2), rTHOuter, inner_radius=rTHInner,
                                 initial_angle=-135 * np.pi / 180,
                                 final_angle=0 * np.pi / 180,
                                 tolerance=curve_tolerance), 'or', 0)
        self.boolean(gdspy.Round((aHMount / 2, -aHMount / 2), rTHOuter, inner_radius=rTHInner,
                                 initial_angle=0 * np.pi / 180,
                                 final_angle=135 * np.pi / 180,
                                 tolerance=curve_tolerance), 'or', 0)
        self.boolean(
            gdspy.Rectangle((aHMount / 2 + rTHOuter, aHMount / 2), (aHMount / 2 + rTHInner, aHMount / 2 + rTHOuter)),
            'or', 0)
        self.boolean(
            gdspy.Rectangle((aHMount / 2 + rTHOuter, -aHMount / 2), (aHMount / 2 + rTHInner, -aHMount / 2 - rTHOuter)),
            'or', 0)

        self.boolean(gdspy.Rectangle((-aHMount / 2 - rTHOuter, aHMount / 2 + rTHOuter),
                                     (aHMount / 2 + rTHOuter, aHMount / 2 + rTHInner)), 'or', 0)
        self.boolean(gdspy.Rectangle((-aHMount / 2 - rTHOuter, - aHMount / 2 - rTHOuter),
                                     (aHMount / 2 + rTHOuter, - aHMount / 2 - rTHInner)), 'or', 0)
        self.boolean(gdspy.Rectangle((-aHMount / 2 - rTHOuter, -aHMount / 2 - rTHOuter),
                                     (-aHMount / 2 - rTHInner, aHMount / 2 + rTHOuter)), 'or', 0)

        self.boolean(gdspy.Round((0, 0), aHMount / 2 * np.sqrt(2) - rTHInner,
                                 inner_radius=aHMount / 2 * np.sqrt(2) - rTHOuter,
                                 initial_angle=45 * np.pi / 180,
                                 final_angle=315 * np.pi / 180), 'or', 0)

        # задаем порты
        self.addPort(position=(length, 0), angle=0.)


class PCB_CHIP(Geometry):
    # по идее в подобных дочерних классах должна быть только их инициализация, остольные функции прописаны в классе-родителе. Возможно это можно сделать еще более просто.
    def __init__(self, leftVia=True, rightVia=True):
        super().__init__()
        self.name = 'mounting place for chip'

        # Рисование происходит здесь
        for i in [0, 1, 2]:
            self.isInverted[i] = True

        # сквозные отверстия
        s = 1000
        w = 330
        gap = 300

        distVia = 400
        nViaLen = 4
        rViaT = 300 / 2

        length = rViaT - 100 + (distVia + rViaT * 2) * (nViaLen - 1)
        # print(length)

        # копланар
        cpw = gdspy.Round((0, 0), s / 2 + w, tolerance=curve_tolerance)
        cpw = gdspy.boolean(cpw, gdspy.Rectangle((0, s / 2 + w), (length, -s / 2 - w)), 'or')
        cpw = gdspy.boolean(cpw, gdspy.Round((0, 0), s / 2, tolerance=curve_tolerance), 'not')
        cpw = gdspy.boolean(cpw, gdspy.Rectangle((0, s / 2), (length, -s / 2)), 'not')
        self.boolean(cpw, 'or', 0)

        # переходные отверстия
        self.boolean(gdspy.Round((0, 0), 200 / 2, tolerance=curve_tolerance), 'or', 3)
        self.boolean(gdspy.Round((500 + 200, 0), 200 / 2, tolerance=curve_tolerance), 'or', 3)

        # переход
        sb = 500
        wb = 580
        gap = 300
        cpwb = gdspy.Round((500 + 200, 0), sb / 2 + wb, tolerance=curve_tolerance)
        cpwb = gdspy.boolean(cpwb, gdspy.Rectangle((-sb / 2 - 300, sb / 2 + wb), (500 + 200, -sb / 2 - wb)), 'or')

        cpwb = gdspy.boolean(cpwb, gdspy.Round((500 + 200, 0), sb / 2, tolerance=curve_tolerance), 'not')
        cpwb = gdspy.boolean(cpwb, gdspy.Round((0, 0), sb / 2, tolerance=curve_tolerance), 'not')
        cpwb = gdspy.boolean(cpwb, gdspy.Rectangle((0, sb / 2), (500 + 200, -sb / 2)), 'not')
        self.boolean(cpwb, 'or', 2)

        # сквозные отверстия с боков

        if leftVia:
            for i in range(0, nViaLen):
                singleVia = gdspy.Round((rViaT - 100 + (2 * rViaT + distVia) * i, s / 2 + w + gap + rViaT), rViaT,
                                        tolerance=curve_tolerance)
                self.boolean(singleVia, 'or', 3)

        if rightVia:
            for i in range(0, nViaLen):
                singleVia = gdspy.Round((rViaT - 100 + (2 * rViaT + distVia) * i, -s / 2 - w - gap - rViaT), rViaT,
                                        tolerance=curve_tolerance)
                self.boolean(singleVia, 'or', 3)

        # задаем порты
        self.addPort(position=(length, 0), angle=0.)
        self.addPort(position=(-sb / 2, 0), angle=180 * np.pi / 180)


class ChipHole(Geometry):
    # по идее в подобных дочерних классах должна быть только их инициализация, остольные функции прописаны в классе-родителе. Возможно это можно сделать еще более просто.
    def __init__(self, a=10e3, TH=500, r = 1000/2, portsPerEdge=3, portDisp=2000, throughAll=False):
        super().__init__()
        self.name = 'Hole under chip'

        # Рисование происходит здесь
        for i in [0, 1, 2]:
            self.isInverted[i] = True

        if throughAll:
            self.boolean(gdspy.Rectangle((-a / 2, -a / 2), (a / 2, a / 2)).fillet(r), 'or', 1)
            self.boolean(gdspy.Rectangle((-a / 2 - TH, -a / 2 - TH), (a / 2 + TH, a / 2 + TH)).fillet(r + TH), 'or', 0)
            self.boolean(gdspy.Rectangle((-a / 2 - TH, -a / 2 - TH), (a / 2 + TH, a / 2 + TH)).fillet(r + TH), 'or', 2)
        else:
            self.boolean(gdspy.Rectangle((-a / 2, -a / 2), (a / 2, a / 2)).fillet(r), 'or', 6)
            self.boolean(gdspy.Rectangle((-a / 2 - TH, -a / 2 - TH), (a / 2 + TH, a / 2 + TH)).fillet(r + TH), 'or', 2)

        # задаем порты
        if portsPerEdge % 2 == 1:
            initPosition = np.array([a / 2 + TH, -portsPerEdge // 2 * portDisp])
        else:
            initPosition = np.array([a / 2 + TH, -portsPerEdge // 2 * portDisp  + portDisp/2])

        for i in range(0, portsPerEdge):
            self.addPort(position=initPosition + np.array([0, portDisp]) * i, angle=0.)

        for i in range(0, portsPerEdge):
            self.addPort(position=np.dot(np.array([[0, -1], [1, 0]]), (initPosition + np.array([0, portDisp]) * i)),
                         angle=90 * np.pi / 180)

        for i in range(0, portsPerEdge):
            self.addPort(position=np.dot(np.array([[-1, 0], [0, -1]]), (initPosition + np.array([0, portDisp]) * i)),
                         angle=180 * np.pi / 180)

        for i in range(0, portsPerEdge):
            self.addPort(position=np.dot(np.array([[0, 1], [-1, 0]]), (initPosition + np.array([0, portDisp]) * i)),
                         angle=270 * np.pi / 180)


class ThroughHole(Geometry):
    # по идее в подобных дочерних классах должна быть только их инициализация, остольные функции прописаны в классе-родителе. Возможно это можно сделать еще более просто.
    def __init__(self, d):
        super().__init__()
        self.name = 'Through hole'

        # Рисование происходит здесь
        for i in [0, 1, 2]:
            self.isInverted[i] = True

        for i in [0, 1, 2]:
            self.boolean(gdspy.Round((0, 0), d/2, tolerance=curve_tolerance), 'or', i)
