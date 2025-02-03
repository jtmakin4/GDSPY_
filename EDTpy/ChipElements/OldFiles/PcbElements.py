from BaseClasses import *


class CPW(Geometry):
    # копланары должны создаваться уже внутри скетча,  но в принципе можно задавать как хочешь
    def __init__(self, port1, port2, s, w, r=None, gap=300, inter_dist=300, via_r=300/2, path=None, chessOrder=False):
        # Оставил возможность задавать свою траекторию
        super().__init__(name='cpw')
        self.isInverted[0] = True
        self.isInverted[2] = True
        self.port1 = port1
        self.port2 = port2

        self._s = s
        self._w = w
        self._g = gap
        if inter_dist is not None:
            self.inter_dist = inter_dist + via_r*2
        else:
            self.inter_dist = None
        self.via_r = via_r

        self._path = path
        self._r = r
        self.length = None

        # Нахождение path по алгоритму, или значением, заданным пользователем
        if self._path is None:
            self._path, self.Rmax = self.getTrajectory()
            if self.Rmax is not None:
                self.Rmax = round(self.Rmax / 1000, 1) * 1000 * 0.99

            if self._r is None:
                self._r = self.Rmax
            elif self._r > self.Rmax:
                raise ValueError('User radius is too big')
            else:
                pass

        # В случае поломки getTrajectory self._path задается как None, в этом случае это пишется и линия не рисуется
        if self._path is not None:
            self._in_point = self._path[0]
            self._out_point = self._path[-1]

            w_polygon = gdspy.FlexPath(self._path,
                                       self._s + 2 * self._w,
                                       corners="circular bend",
                                       bend_radius=self._r,
                                       tolerance=self.curve_tolerance*0.5,
                                       precision=self.curve_precision)

            if self._r is not None:
                central_line = gdspy.FlexPath(self._path,
                                              self._s*0.01,
                                              corners="circular bend",
                                              bend_radius=self._r,
                                              tolerance=self.curve_tolerance*0.5,
                                              precision=self.curve_precision)
            else:
                central_line = gdspy.FlexPath(self._path,
                                              self._s*0.01,
                                              tolerance=self.curve_tolerance*0.5,
                                              precision=self.curve_precision)

            self.length = round(central_line.area()/0.01/self._s, 0)

            s_polygon = gdspy.FlexPath(self._path,
                                       self._s,
                                       corners="circular bend",
                                       bend_radius=self._r,
                                       tolerance=self.curve_tolerance*0.5,
                                       precision=self.curve_precision)
            w_polygon = gdspy.boolean(w_polygon, s_polygon, 'not')
            self.boolean(w_polygon, 'or', 0)

            # Отрисовка сквозных отверстий
            if self.inter_dist is not None:
                viaNum = self.length / (inter_dist + 2 * via_r)
                viaNum = int(round(viaNum, 0))
                self.inter_dist = self.length / viaNum
                via_polygon = gdspy.PolygonSet([])

                offset = self.via_r + 300 + self._s / 2 + self._w

                via_centers_left, via_centers_right = self._create_via_centers(self._path, self._r, offset, self.inter_dist)

                if not chessOrder:
                    for i in range(0, len(via_centers_left)):
                        arc = gdspy.Round(
                            via_centers_left[i],
                            via_r,
                            tolerance=self.curve_tolerance,
                        )
                        via_polygon = gdspy.boolean(via_polygon, arc, 'or')
                    for i in range(0, len(via_centers_right)):
                        arc = gdspy.Round(
                            via_centers_right[i],
                            via_r,
                            tolerance=self.curve_tolerance,
                        )
                        via_polygon = gdspy.boolean(via_polygon, arc, 'or')
                else:
                    for i in range(0, len(via_centers_left)):
                        if i % 2 == 0:
                            arc = gdspy.Round(
                                via_centers_left[i],
                                via_r,
                                tolerance=self.curve_tolerance,
                            )
                            via_polygon = gdspy.boolean(via_polygon, arc, 'or')
                    for i in range(0, len(via_centers_right)):
                        if i % 2 == 1:
                            arc = gdspy.Round(
                                via_centers_right[i],
                                via_r,
                                tolerance=self.curve_tolerance,
                            )
                            via_polygon = gdspy.boolean(via_polygon, arc, 'or')
                self.boolean(via_polygon, 'or', 3)
                # self.boolean(tech_polygon, 'or', 5)

            print(f'CPW R = {self._r}, length = {round(self.length, 0)} \npath:')
            for i in range(len(self._path)):
                if i == 0:
                    print(f'x = {round(self._path[i][0],2)},\ty = {round(self._path[i][1],2)}\t(port 1)')
                elif i == len(self._path)-1:
                    print(f'x = {round(self._path[i][0],2)},\ty = {round(self._path[i][1],2)}\t(port 2)')
                else:
                    print(f'x = {round(self._path[i][0],2)},\ty = {round(self._path[i][1],2)}')



    def getTrajectory(self):
        s1 = None
        s2 = None
        a1, o1 = self.port1.basis()
        r1 = np.array(self.port1.position)
        a2, o2 = self.port2.basis()
        r2 = np.array(self.port2.position)
        delt_r = r1 - r2

        # Порты смотрят в одну сторону (по идее надо сделать еще проверку на то, что смотрит в разные стороны, но я не зваю как это сделать)
        if np.dot(a1, a2) > 0:
            print('Can not draw path for waveguide')
            return None, None

        # Опрределяем направление на центр окружностей (s - sign знак для вектора o портов)
        if np.dot(-delt_r, o1) > 0:
            s1 = 1
        if np.dot(-delt_r, o1) < 0:
            s1 = -1
        if abs(np.dot(delt_r, o2)) < 1e-6:
            s1 = 0

        if np.dot(delt_r, o2) > 0:
            s2 = 1
        if np.dot(delt_r, o2) < 0:
            s2 = -1
        if abs(np.dot(delt_r, o2)) < 1e-6:
            s2 = 0

        """
        Далее рассмотриены различные конфигурации линий
        """

        # Порты находятся на одной линии
        if s1 == 0 and s2 == 0:
            path = [r1,  r2]
            Rmax = None
            return path, Rmax

        # Один из портов смотрит на другой
        if (s1 == 0 and s2 != 0) or (s1 != 0 and s2 == 0):
            if s1 == 0 and s2 != 0:
                s1 = 1
            else:
                s2 = 1

            delt_o = s1 * o1 - s2 * o2
            D = np.dot(delt_o, delt_r) ** 2
            D += 2 * np.dot(delt_r, delt_r) * (1 + s1 * s2 * np.dot(o1, o2))
            Rmax = np.dot(delt_o, delt_r) + np.sqrt(D)
            Rmax /= (2 * (1 + s1 * s2 * np.dot(o1, o2)))

            c1 = r1 + Rmax * s1 * o1
            c2 = r2 + Rmax * s2 * o2
            delt_c = c1 - c2
            p = (c1 + c2) / 2

            # Создаем промужуточные точки на кривой
            x1 = np.dot(p - r1, delt_c) / np.dot(a1, delt_c)
            x2 = np.dot(p - r2, delt_c) / np.dot(a2, delt_c)

            path = [r1, r1 + a1 * x1, r2 + a2 * x2, r2]
            return path, Rmax

        # Порты друг на друга не смотрят, лучи, исходящие из портов, не пересекаются
        if s1*s2 > 0:
            if (1 + s1 * s2 * np.dot(o1, o2)) > 1e-9:
                delt_o = s1 * o1 - s2 * o2
                D = np.dot(delt_o, delt_r) ** 2
                D += 2 * np.dot(delt_r, delt_r) * (1 + s1 * s2 * np.dot(o1, o2))
                Rmax = np.dot(delt_o, delt_r) + np.sqrt(D)
                Rmax /= (2 * (1 + s1 * s2 * np.dot(o1, o2)))

                c1 = r1 + Rmax * s1 * o1
                c2 = r2 + Rmax * s2 * o2
                delt_c = c1 - c2
                p = (c1 + c2) / 2
                # Создаем промужуточные точки на кривой
                x1 = np.dot(p - r1, delt_c) / np.dot(a1, delt_c)
                x2 = np.dot(p - r2, delt_c) / np.dot(a2, delt_c)

                path = [r1, r1 + a1 * x1, r2 + a2 * x2, r2]
                return path, Rmax
            else:
                # Направление портов параллельно
                angle = np.arccos(np.dot(delt_r, a2)/np.sqrt(np.dot(delt_r, delt_r)))
                t = np.sqrt(np.dot(delt_r, delt_r))/4/np.cos(angle)
                Rmax = t/np.tan(angle)

                path = [r1, r1 + a1 * t, r2 + a2 * t, r2]
                return path, Rmax

        # Порты друг на друга не смотрят, лучи, исходящие из портов, пересекаются
        if s1*s2 < 0:
            A = np.vstack((-a1, a2))
            t1, t2 = np.linalg.solve(A.transpose(), delt_r)
            r3 = r1+a1*t1 # точка пересечения прямых, выходящих из портов
            a3 = -a1-a2
            a3 = a3/np.sqrt(np.dot(a3, a3)) # биссектрисса

            if t1 > t2:
                M = np.vstack((o2*s2, -a3))
                Rmax = abs(np.linalg.solve(M.transpose(), r3-r2)[1])
            else:
                M = np.vstack((o1*s1, -a3))
                Rmax = abs(np.linalg.solve(M.transpose(), r3-r1)[1])

            path = [r1, r3, r2]
            return path, Rmax

    # Старые функции для нахождения положения сквозных отверстий
    # Здесь не обработан случай, когда кривая состтоит только из окружностей
    # Для обхода этого Радиус уменьшен на 1 %
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

        return points_array_left[1:], points_array_right[1:]


class SMP_PCB(Geometry):
    # по идее в подобных дочерних классах должна быть только их инициализация, остольные функции прописаны в классе-родителе. Возможно это можно сделать еще более просто.
    def __init__(self, soldering=True):
        super().__init__()
        self.name = 'mounting place for SMP-PCB connector'
        self._isSoldering = soldering
        # Рисование происходит здесь
        for i in [0, 1, 2]:
            self.isInverted[i] = True

        # сквозные отверстия
        nViaT = 18
        rViaT = 300 / 2
        rViaPosT = 2000
        s = 1000
        w = 330
        gap = 300

        startAngle = np.arcsin((rViaT + +s / 2 + w + gap) / rViaPosT)
        singleViaAng = (np.pi * 2 - 2 * startAngle) / (nViaT - 1)

        singleVia = gdspy.Round((rViaPosT * np.cos(startAngle), rViaPosT * np.sin(startAngle)), rViaT,
                                tolerance=self.curve_tolerance)

        for i in range(0, nViaT):
            self.boolean(singleVia, 'or', 3)
            singleVia.rotate(singleViaAng)

        rs = 1050 / 2
        rw = rs + 600

        distVia = 300
        nViaLen = 3
        length = rViaPosT * np.cos(startAngle) + (distVia + 2 * rViaT) * nViaLen

        singleViaDown = gdspy.Round((rViaPosT * np.cos(startAngle), -rViaPosT * np.sin(startAngle)), rViaT,
                                    tolerance=self.curve_tolerance)
        singleViaUp = gdspy.Round((rViaPosT * np.cos(startAngle), rViaPosT * np.sin(startAngle)), rViaT,
                                  tolerance=self.curve_tolerance)
        for i in range(0, nViaLen):
            singleViaDown.translate(distVia + 2 * rViaT, 0)
            self.boolean(singleViaDown, 'or', 3)
            singleViaUp.translate(distVia + 2 * rViaT, 0)
            self.boolean(singleViaUp, 'or', 3)

        # разъем со стороны копланаров
        cpw = gdspy.Round((0, 0), rw, tolerance=self.curve_tolerance)
        cpw = gdspy.boolean(cpw, gdspy.Rectangle((0, s / 2 + w), (length, -s / 2 - w)), 'or')
        cpw = gdspy.boolean(cpw, gdspy.Round((0, 0), rs, tolerance=self.curve_tolerance), 'not')
        cpw = gdspy.boolean(cpw, gdspy.Rectangle((0, s / 2), (length, -s / 2)), 'not')

        self.boolean(cpw, 'or', 0)

        # разъем со стороны коннектора
        rsb = 1100 / 2
        rwb = rs + 930
        self.boolean(gdspy.Round((0, 0), rwb, inner_radius=rsb, tolerance=self.curve_tolerance), 'or', 2)

        # центральное отверстие
        rhole = 800 / 2
        for i in [0, 1, 2]:
            self.boolean(gdspy.Round((0, 0), rhole, tolerance=self.curve_tolerance), 'or', i)

        # крепления
        if self._isSoldering:
            rHMount = 1000 / 2
            aHMount = 5080
            rTHInner = rHMount + 200
            rTHOuter = rTHInner + 200

            for i in [-1, 1]:
                for j in [-1, 1]:
                    for k in [0, 1, 2]:
                        self.boolean(gdspy.Round((i * aHMount / 2, j * aHMount / 2), rHMount, tolerance=self.curve_tolerance),
                                     'or', k)

            self.boolean(gdspy.Round((aHMount / 2, aHMount / 2), rTHOuter, inner_radius=rTHInner,
                                     initial_angle=-135 * np.pi / 180,
                                     final_angle=0 * np.pi / 180,
                                     tolerance=self.curve_tolerance), 'or', 0)
            self.boolean(gdspy.Round((aHMount / 2, -aHMount / 2), rTHOuter, inner_radius=rTHInner,
                                     initial_angle=0 * np.pi / 180,
                                     final_angle=135 * np.pi / 180,
                                     tolerance=self.curve_tolerance), 'or', 0)
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
                                     final_angle=315 * np.pi / 180,
                                     tolerance=self.curve_tolerance), 'or', 0)

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
        cpw = gdspy.Round((0, 0), s / 2 + w, tolerance=self.curve_tolerance)
        cpw = gdspy.boolean(cpw, gdspy.Rectangle((0, s / 2 + w), (length, -s / 2 - w)), 'or')
        cpw = gdspy.boolean(cpw, gdspy.Round((0, 0), s / 2, tolerance=self.curve_tolerance), 'not')
        cpw = gdspy.boolean(cpw, gdspy.Rectangle((0, s / 2), (length, -s / 2)), 'not')
        self.boolean(cpw, 'or', 0)

        # переходные отверстия
        self.boolean(gdspy.Round((0, 0), 200 / 2, tolerance=self.curve_tolerance), 'or', 3)
        self.boolean(gdspy.Round((500 + 200, 0), 200 / 2, tolerance=self.curve_tolerance), 'or', 3)

        # переход
        sb = 500
        wb = 580
        gap = 300
        cpwb = gdspy.Round((500 + 200, 0), sb / 2 + wb, tolerance=self.curve_tolerance)
        cpwb = gdspy.boolean(cpwb, gdspy.Rectangle((-sb / 2 - 500-700, sb / 2 + wb), (500 + 200, -sb / 2 - wb)), 'or')

        cpwb = gdspy.boolean(cpwb, gdspy.Round((500 + 200, 0), sb / 2, tolerance=self.curve_tolerance), 'not')
        cpwb = gdspy.boolean(cpwb, gdspy.Round((-700, 0), sb / 2, tolerance=self.curve_tolerance), 'not')
        cpwb = gdspy.boolean(cpwb, gdspy.Rectangle((-700, sb / 2), (500 + 200, -sb / 2)), 'not')
        self.boolean(cpwb, 'or', 2)

        # сквозные отверстия с боков

        if leftVia:
            for i in range(0, nViaLen):
                singleVia = gdspy.Round((rViaT - 100 + (2 * rViaT + distVia) * i, s / 2 + w + gap + rViaT), rViaT,
                                        tolerance=self.curve_tolerance)
                self.boolean(singleVia, 'or', 3)

        if rightVia:
            for i in range(0, nViaLen):
                singleVia = gdspy.Round((rViaT - 100 + (2 * rViaT + distVia) * i, -s / 2 - w - gap - rViaT), rViaT,
                                        tolerance=self.curve_tolerance)
                self.boolean(singleVia, 'or', 3)

        # задаем порты
        self.addPort(position=(length, 0), angle=0.)
        self.addPort(position=(-sb / 2-700, 0), angle=180 * np.pi / 180)


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
            self.boolean(gdspy.Rectangle((-a / 2 - 300, -a / 2 - 300), (a / 2 + 300, a / 2 + 300)).fillet(r + 300), 'or', 0)
            self.boolean(gdspy.Rectangle((-a / 2 - TH, -a / 2 - TH), (a / 2 + TH, a / 2 + TH)).fillet(r + TH), 'or', 2)

            self.boolean(gdspy.Round(center=(-a / 2 + r/np.sqrt(2), -a / 2 + r/np.sqrt(2)), radius=r, tolerance=self.curve_tolerance), 'or', 1)
            self.boolean(gdspy.Round(center=(-a / 2 + r/np.sqrt(2), a / 2 - r/np.sqrt(2)), radius=r, tolerance=self.curve_tolerance), 'or', 1)
            self.boolean(gdspy.Round(center=(a / 2 - r/np.sqrt(2), -a / 2 + r/np.sqrt(2)), radius=r, tolerance=self.curve_tolerance), 'or', 1)
            self.boolean(gdspy.Round(center=(a / 2 - r/np.sqrt(2), a / 2 - r/np.sqrt(2)), radius=r, tolerance=self.curve_tolerance), 'or', 1)

            self.boolean(gdspy.Round(center=(-a / 2 + r/np.sqrt(2), -a / 2 + r/np.sqrt(2)), radius=r + 300, tolerance=self.curve_tolerance), 'or', 2)
            self.boolean(gdspy.Round(center=(-a / 2 + r/np.sqrt(2), a / 2 - r/np.sqrt(2)), radius=r + 300, tolerance=self.curve_tolerance), 'or', 2)
            self.boolean(gdspy.Round(center=(a / 2 - r/np.sqrt(2), -a / 2 + r/np.sqrt(2)), radius=r + 300, tolerance=self.curve_tolerance), 'or', 2)
            self.boolean(gdspy.Round(center=(a / 2 - r/np.sqrt(2), a / 2 - r/np.sqrt(2)), radius=r + 300, tolerance=self.curve_tolerance), 'or', 2)

            self.boolean(gdspy.Round(center=(-a / 2 + r/np.sqrt(2), -a / 2 + r/np.sqrt(2)), radius=r + 300, tolerance=self.curve_tolerance), 'or', 0)
            self.boolean(gdspy.Round(center=(-a / 2 + r/np.sqrt(2), a / 2 - r/np.sqrt(2)), radius=r + 300, tolerance=self.curve_tolerance), 'or', 0)
            self.boolean(gdspy.Round(center=(a / 2 - r/np.sqrt(2), -a / 2 + r/np.sqrt(2)), radius=r + 300, tolerance=self.curve_tolerance), 'or', 0)
            self.boolean(gdspy.Round(center=(a / 2 - r/np.sqrt(2), a / 2 - r/np.sqrt(2)), radius=r + 300, tolerance=self.curve_tolerance), 'or', 0)
        else:
            self.boolean(gdspy.Rectangle((-a / 2, -a / 2), (a / 2, a / 2)).fillet(r), 'or', 6)
            self.boolean(gdspy.Rectangle((-a / 2 - TH, -a / 2 - TH), (a / 2 + TH, a / 2 + TH)).fillet(r + TH), 'or', 2)

        # задаем порты
        if portsPerEdge % 2 == 1:
            initPosition = np.array([a / 2 + TH, -(portsPerEdge-1) // 2 * portDisp])
        else:
            initPosition = np.array([a / 2 + TH, -portsPerEdge // 2 * portDisp + portDisp/2])

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
            self.boolean(gdspy.Round((0, 0), d/2, tolerance=self.curve_tolerance), 'or', i)
