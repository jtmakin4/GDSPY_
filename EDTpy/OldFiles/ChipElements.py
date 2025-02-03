from BaseClasses import *


class SawMark(Geometry):
    def __init__(self, position=(0, 0)):
        super().__init__(name='SawMark', unit=1e-6, precision=1e-9)
        self.curve_tolerance = 0.1
        self.curve_precision = 0.1

        self.boolean(gdspy.Rectangle((-600/2, -600/2), (600/2, 600/2)), 'or', 0)

        for i in range(0, 4):
            rect = gdspy.Rectangle((-600 / 2, -600 / 2), (-600/2+106.5, -600/2+106.5))
            rect.rotate(90/180*np.pi*i)
            self.boolean(rect, 'not', 0)

            rect = gdspy.Rectangle((-250, -2.5), (-10, 2.5))
            rect.rotate(1/2 * np.pi * i)
            self.boolean(rect, 'not', 0)
        for i in range(0, 2):
            self.boolean(gdspy.Rectangle((-30 / 2, -2 / 2), (30 / 2, 2 / 2)).rotate(90/180*np.pi*i), 'not', 0)

        self.move(position)


class LitographyMark(Geometry):
    def __init__(self, position=(0, 0)):
        super().__init__(name='LitographyMask', unit=1e-6, precision=1e-9)
        self.curve_tolerance = 0.1
        self.curve_precision = 0.1

        self.boolean(gdspy.Round(center=(0, 0), radius=170/2, max_points = 1000), 'or', 0)
        self.boolean(gdspy.Round(center=(0, 0), radius=370/2, inner_radius=200/2, max_points = 1000), 'or', 0)
        self.boolean(gdspy.Round(center=(0, 0), radius=430/2, inner_radius=400/2, max_points = 1000), 'or', 0)
        self.boolean(gdspy.Rectangle((-380/2, -4/2), (380/2, 4/2)), 'not', 0)
        self.boolean(gdspy.Rectangle((-4/2, -380/2), (4/2, 380/2)), 'not', 0)
        self.boolean(gdspy.Rectangle((-2/2, -2/2), (2/2, 2/2)).rotate(45*np.pi/180), 'or', 0)

        self.move(position)


class ChipCPWPort(Geometry):
    def __init__(self, base_length, translation_length, S, W, S_PCB = 500, W_PCB = 580, smothing_evaluations=5*6):
        super().__init__(name='ChipPort', unit=1e-6, precision=1e-9)
        self.curve_tolerance = 0.1
        self.curve_precision = 0.1

        path = gdspy.Path(S_PCB+W_PCB*2, (0, 0))
        path.segment(base_length, '+x')
        path.parametric(lambda u: np.array((u*translation_length, 0)),
                        final_width=lambda u: (-S-W*2 + S_PCB+W_PCB*2)*np.cos(np.pi*u)/2 + (S+W*2 + S_PCB+W_PCB*2)/2,
                        number_of_evaluations=smothing_evaluations,
                        tolerance=self.curve_tolerance)
        self.boolean(path, 'or', 0)

        path = gdspy.Path(S_PCB, (0, 0))
        path.segment(base_length, '+x')
        path.parametric(lambda u: np.array((u * translation_length, 0)),
                        final_width=lambda u: (-S + S_PCB) * np.cos(np.pi * u) / 2 + (
                                    S + S_PCB) / 2,
                        number_of_evaluations=smothing_evaluations,
                        tolerance=self.curve_tolerance)
        self.boolean(path, 'not', 0)

        self.addPort(position=(0, 0), angle=np.pi)
        self.addPort(position=(base_length + translation_length, 0), angle=0)

# todo исправить бак с определением радиуса для поворота в одну сторону, наблюдался при построении ыипа, когда требовалось сильно изогнуть линию
class CPW_Chip(Geometry):
    def __init__(self, port1, port2, s, w, r=None, path=None):
        super().__init__(name='cpw')
        self.curve_tolerance = 0.1
        self.curve_precision = 0.1

        self.port1 = port1
        self.port2 = port2

        self._s = s
        self._w = w

        self._path = path
        self._r = r
        self.length = None

        # Нахождение path по алгоритму, или значением, заданным пользователем
        if self._path is None:
            self._path, self.Rmax = self.getTrajectory()
            if self.Rmax is not None:
                self.Rmax = round(self.Rmax / 1000, 1) * 1000 * 0.99

            if self.Rmax is not None:
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
            # определяем длину
            if self._r is not None:
                central_line = gdspy.FlexPath(self._path,
                                              1,
                                              corners="circular bend",
                                              bend_radius=self._r,
                                              tolerance=0.001,
                                              precision=0.001)
            else:
                central_line = gdspy.FlexPath(self._path,
                                              1,
                                              tolerance=0.001,
                                              precision=0.001)

            self.length = round(central_line.area(), 3)

            s_polygon = gdspy.FlexPath(self._path,
                                       self._s,
                                       corners="circular bend",
                                       bend_radius=self._r,
                                       tolerance=self.curve_tolerance*0.5,
                                       precision=self.curve_precision)
            w_polygon = gdspy.boolean(w_polygon, s_polygon, 'not')
            self.boolean(w_polygon, 'or', 0)

            print(f'CPW R = {self._r}, length = {round(self.length, 0)} \npath:')
            # for i in range(len(self._path)):
            #     if i == 0:
            #         print(f'x = {round(self._path[i][0],2)},\ty = {round(self._path[i][1],2)}\t(port 1)')
            #     elif i == len(self._path)-1:
            #         print(f'x = {round(self._path[i][0],2)},\ty = {round(self._path[i][1],2)}\t(port 2)')
            #     else:
            #         print(f'x = {round(self._path[i][0],2)},\ty = {round(self._path[i][1],2)}')

    def getTrajectory(self):
        s1 = None
        s2 = None
        a1, o1 = self.port1.basis()
        r1 = np.array(self.port1.position)
        a2, o2 = self.port2.basis()
        r2 = np.array(self.port2.position)
        delt_r = r1 - r2

        # Порты смотрят в одну сторону (по идее надо сделать еще проверку на то, что смотрит в разные стороны, но я не зваю как это сделать)
        # здесь исправлена ошибка округления
        if np.dot(a1, a2) > 1e-6:
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


class BiasEnd(Geometry):
    def __init__(self, S = 11, W = 2, gap_asymerty=15, position='L'):
        super().__init__(name='ViasEnd', unit=1e-6, precision=1e-9)
        self.curve_tolerance = 0.1
        self.curve_precision = 0.1
        self.addPort(position=(0, 0), angle=3 * np.pi / 2)
        self.addPort(position=(0, gap_asymerty), angle=np.pi / 2)
        if position == 'L':
            self.boolean(gdspy.Rectangle((W+S/2, 0), (S/2, gap_asymerty)), 'or', 0)
        elif position == 'R':
            self.boolean(gdspy.Rectangle((-W - S / 2, 0), (-S / 2, gap_asymerty)), 'or', 0)
        else:
            raise ValueError('position should be L or R')


class ACEnd(Geometry):
    def __init__(self, S = 11, W = 2, length = 2):
        super().__init__(name='ViasEnd', unit=1e-6, precision=1e-9)
        self.curve_tolerance = 0.1
        self.curve_precision = 0.1
        self.addPort(position=(0, 0), angle=3 * np.pi / 2)
        self.addPort(position=(0, length), angle=np.pi / 2)
        self.boolean(gdspy.Rectangle((W + S / 2, 0), (-W - S / 2, length)), 'or', 0)


class NotchQuarterCavity(Geometry):
    def __init__(self, position=(0, 0), angle=0, freq=6e9, S=21.2, W=8.8, S_feed=25.2, W_feed=10.8, gap=15.2, interLen=400,
                 innerGap = None, n=4, r=60, neck=200, tail=None):
        super().__init__(name='NotchQuarterCavity', unit=1e-6, precision=1e-9)
        self.curve_tolerance = 0.1
        self.curve_precision = 0.1

        b = 2 * S + W
        if innerGap is None:
            innerGap = 4*b

        # порты входа и выхода
        port1 = Port((-interLen / 2 - innerGap / 2 - innerGap, 0), 0)
        port2 = Port((+interLen / 2 + innerGap / 2 + innerGap, 0), np.pi)
        feedline = CPW_Chip(port1, port2, S_feed, W_feed).elements[0]
        self.boolean(feedline, 'or', 0)
        port1.rotate(np.pi)
        port2.rotate(np.pi)
        self.addPort(position=(-interLen / 2 - innerGap / 2 - innerGap, 0), angle=np.pi)
        self.addPort(position=(+interLen / 2 + innerGap / 2 + innerGap, 0), angle=0)

        # отрисовка базы для резонатора
        base_v = np.array([0, b/2 + gap + S_feed/2+W_feed])
        path = []
        path += [tuple(np.array([0, 2*r*2*n + innerGap + neck]) + base_v)]
        path += [tuple(np.array(path[-1]) + np.array([0, -neck]))]
        path += [tuple(np.array(path[-1]) + np.array([interLen / 2 + innerGap / 2, 0]))]
        path += [tuple(base_v + np.array([interLen / 2 + innerGap / 2, 0]))]
        path += [tuple(np.array(path[-1]) + np.array([-innerGap, 0]))]
        for i in range(0, n):
            path += [tuple(np.array(path[-1]) + np.array([-(interLen-innerGap), 0]))]
            path += [tuple(np.array(path[-1]) + np.array([0, 2 * r]))]
            path += [tuple(np.array(path[-1]) + np.array([(interLen-innerGap), 0]))]
            path += [tuple(np.array(path[-1]) + np.array([0, 2 * r]))]

        path += [tuple(np.array(path[-1]) + np.array([-interLen, 0]))]

        # дорисовка хвоста для согласования частот
        path += [tuple(np.array(path[-1]) + np.array([0, -r]))]
        if tail is not None:
            path += [tuple(np.array(path[-1]) + np.array([0, -tail]))]
        else:
            pass # дописать подконку под частоту

        resonator = CPW_Chip(None, None, S, W, r=r, path=path)
        # длина ломаной
        length = neck + (interLen / 2 + innerGap / 2) + (4*r*n + innerGap) + 2*interLen +\
                 (2*r + interLen - innerGap)*2*n - (interLen - innerGap) + r
        if tail is not None:
            length += tail
        # компенсация на скруглениях
        length -= (4*n+4)*(2*r-np.pi*r/2)
        print(length)
        self.boolean(resonator.elements[0], 'or', 0)
        self.addPort(position=path[0], angle=np.pi/2)
        print(f'heidth = {(np.array(path[0]) - base_v)[1] + b/2}, width = {interLen + innerGap}')


class Cross(Geometry):
    def __init__(self, a=450, b=400, S = 61.2, W = 18.8, S_claw = 21.2, W_claw = 8.8, S_drive = 12.2, W_drive = 4.5, clawLen = 30, clawDisp = 18.8, clawWidth=12):
        super().__init__(name='NotchQuarterCavity', unit=1e-6, precision=1e-9)
        self.curve_tolerance = 0.1
        self.curve_precision = 0.1
        self.addPort(position=(0, 0), angle=3*np.pi/2)

        groundOffset = 40
        self.boolean(gdspy.Rectangle((-S/2-clawDisp-clawWidth-groundOffset, 0),
                                     (S/2+clawDisp+clawWidth+groundOffset, 3*groundOffset+clawWidth + clawLen)), 'or', 0)
        cross_center = np.array([0, groundOffset*3 + clawWidth + clawLen + b/2])
        cross_top = cross_center + np.array([0, b/2])
        self.boolean(gdspy.Rectangle(tuple(cross_center+np.array([-S/2-W, -b/2-groundOffset - clawLen])),
                                     tuple(cross_center+np.array([S/2+W, b/2]))), 'or', 0)
        self.boolean(gdspy.Rectangle(tuple(cross_center+np.array([-a/2, -S/2-W])),
                                     tuple(cross_center+np.array([a/2, S/2+W]))), 'or', 0)

        cross = gdspy.Rectangle(tuple(cross_center+np.array([-S/2, -b/2-groundOffset - clawLen])),
                                     tuple(cross_center+np.array([S/2, b/2-W])))
        cross = gdspy.boolean(cross,
                              gdspy.Rectangle(
                                  tuple(cross_center + np.array([-a/2+W, -S/2])),
                                  tuple(cross_center + np.array([a/2-W, S/2]))),
                              'or')



        claw = gdspy.Rectangle((-S_claw/2, 0), (S_claw/2, groundOffset))
        claw = gdspy.boolean(claw, gdspy.Rectangle((-S/2-clawDisp-clawWidth, groundOffset),
                                                   (S/2+clawDisp+clawWidth, groundOffset+clawWidth+groundOffset+clawLen)), 'or')
        claw = gdspy.boolean(claw, gdspy.Rectangle((-S/2-clawDisp, groundOffset+clawWidth),
                                                   (S/2+clawDisp, groundOffset+clawWidth+groundOffset+clawLen)), 'not')
        self.boolean(gdspy.boolean(claw, cross, 'or'), 'not', 0)

        # разъем для сквида
        self.addPort(position=tuple(cross_center + np.array([0, b/2-W])), angle=np.pi/2)

        # bias линия
        self.addPort(position=tuple(cross_top + np.array([10, 15])), angle=np.pi / 2)

        # ac линия
        self.addPort(position=tuple(cross_top + np.array([a/2 + 100, -180])), angle=np.pi / 2)


class SQUID_3point(Geometry):
    def __init__(self, h1=0.25, w1=0.3, h2=0.115, w2=0.115, h=18.8):
        super().__init__(name='NotchQuarterCavity', unit=1e-6, precision=1e-9)
        self.curve_tolerance = 0.005
        self.curve_precision = 0.005

        squid_layer = 1

        # параметры падов, крепящихся к островкам металлизации
        bottom_pad_width = 8
        bottom_pad_height = 13
        top_pad_width = 8
        top_pad_height = 13

        # параметры вилки
        fork_dist = 8.5
        bottom_leg_len = 0.72

        # расстояние между границами нижних падов
        interpad_dist = 4

        # расстояние между верхними и нижними ножками
        dist = 0.2

        # выход слоя сквида за острова
        treshhold = 2.5

        # ножки сквида
        base_width = 1
        precise_width = base_width/2
        top_len = [1.5, 2.4] # первое значение для базы второе для точной толщины
        top_leg_len = 2
        bottom_len = 1.68


        # параметры бондажа
        bondage_width = 7.5
        bondage_height = 12
        bondage_r = 1.8
        bandage_layer = 2

        # параметры вырезов на граунд плейне
        gp_cut_width = 4
        gp_cut_heidth = 11

        self.addPort(position=(0, 0), angle=3*np.pi/2)

        # рисование падов
        self.boolean(gdspy.Rectangle((-bottom_pad_width/2, 0),
                                     (bottom_pad_width/2, -bottom_pad_height)).translate(dx=0, dy=treshhold), 'or', squid_layer)

        self.boolean(gdspy.Rectangle((-top_pad_width-interpad_dist/2, h),
                                     (-interpad_dist/2, h+top_pad_height)).translate(dx=0, dy=-treshhold), 'or', squid_layer)

        self.boolean(gdspy.Rectangle((-top_pad_width-interpad_dist/2, h),
                                     (-interpad_dist/2, h+top_pad_height)).translate(dx=interpad_dist+top_pad_width, dy=-treshhold), 'or', squid_layer)


        # рисование бондажа
        self.boolean(gdspy.Rectangle((-bondage_width/2, 0),
                                     (bondage_width/2, -bondage_height)).translate(dx=0, dy=treshhold-bottom_pad_height/2).fillet(bondage_r), 'or', bandage_layer)

        self.boolean(gdspy.Rectangle((-bondage_width - (interpad_dist + top_pad_width-bondage_width) / 2, h),
                                     (-interpad_dist / 2 - (top_pad_width-bondage_width) / 2, h + top_pad_height)).translate(dx=0, dy=-treshhold + top_pad_height/2).fillet(bondage_r), 'or', bandage_layer)

        self.boolean(gdspy.Rectangle((-bondage_width - (interpad_dist + top_pad_width-bondage_width) / 2, h),
                                     (-interpad_dist / 2 - (top_pad_width-bondage_width) / 2, h + top_pad_height)).translate(dx=interpad_dist + top_pad_width, dy=-treshhold+ top_pad_height/2).fillet(bondage_r), 'or', bandage_layer)


        # рисование нижних ножек
        self.boolean(gdspy.Rectangle((-base_width - (interpad_dist + top_pad_width - base_width) / 2, h),
                                     (-interpad_dist / 2 - (top_pad_width - base_width) / 2,
                                      h - treshhold - top_len[0])), 'or', squid_layer)

        self.boolean(gdspy.Rectangle((-base_width - (interpad_dist + top_pad_width - base_width) / 2, h),
                                     (-interpad_dist / 2 - (top_pad_width - base_width) / 2,
                                      h - treshhold - top_len[0])).translate(dx=interpad_dist + top_pad_width,
                                                                     dy=0), 'or', squid_layer)

        self.boolean(gdspy.Rectangle((-precise_width - (interpad_dist + top_pad_width - precise_width) / 2, h),
                                     (-interpad_dist / 2 - (top_pad_width - precise_width) / 2,
                                      h - treshhold - top_len[0] - top_len[1])), 'or', squid_layer)

        self.boolean(gdspy.Rectangle((-precise_width - (interpad_dist + top_pad_width - precise_width) / 2, h),
                                     (-interpad_dist / 2 - (top_pad_width - precise_width) / 2,
                                      h - treshhold - top_len[0] - top_len[1])).translate(dx=interpad_dist + top_pad_width,
                                                                             dy=0), 'or', squid_layer)


        # рисование нижних полосок сквида
        self.boolean(gdspy.Rectangle((-interpad_dist/2-top_pad_width/2+precise_width/2, h-treshhold-top_len[0]-top_len[1]),
                                     (-interpad_dist/2-top_pad_width/2+precise_width/2 + top_leg_len, h-treshhold-top_len[0]-top_len[1]+h1)), 'or', squid_layer)

        self.boolean(gdspy.Rectangle((interpad_dist / 2 + top_pad_width / 2 - precise_width / 2, h - treshhold - top_len[0] - top_len[1]),
                                     (interpad_dist / 2 + top_pad_width / 2 - precise_width / 2 - top_leg_len, h - treshhold - top_len[0] - top_len[1] + h2)), 'or', squid_layer)

        # рисование верхних полосок сквида
        self.boolean(gdspy.Rectangle((-fork_dist/2-w1-(-w1+precise_width)/2, h-treshhold-top_len[0]-top_len[1]-dist),
                                     (-fork_dist/2-(-w1+precise_width)/2, h-treshhold-top_len[0]-top_len[1]-dist-bottom_leg_len)), 'or', squid_layer)

        self.boolean(gdspy.Rectangle((fork_dist/2+w2+(-w2+precise_width)/2, h-treshhold-top_len[0]-top_len[1]-dist),
                                     (fork_dist/2+(-w2+precise_width)/2, h-treshhold-top_len[0]-top_len[1]-dist-bottom_leg_len)), 'or', squid_layer)

        # рисование вилки
        self.boolean(gdspy.Rectangle((-fork_dist/2, h-treshhold-top_len[0]-top_len[1]-dist-bottom_leg_len),
                                     (-fork_dist/2-precise_width, h-treshhold-top_len[0]-top_len[1]-dist-bottom_leg_len - bottom_len)), 'or', squid_layer)

        self.boolean(gdspy.Rectangle((fork_dist/2, h-treshhold-top_len[0]-top_len[1]-dist-bottom_leg_len),
                                     (fork_dist/2+precise_width, h-treshhold-top_len[0]-top_len[1]-dist-bottom_leg_len - bottom_len)), 'or', squid_layer)

        self.boolean(gdspy.Rectangle((-fork_dist / 2 - precise_width, h - treshhold - top_len[0] - top_len[1] - dist - bottom_leg_len - bottom_len),
                                     (fork_dist / 2 + precise_width, h - treshhold - top_len[0] - top_len[1] - dist - bottom_leg_len - bottom_len - precise_width)), 'or', squid_layer)

        self.boolean(gdspy.Rectangle((-base_width/2, h - treshhold - top_len[0] - top_len[1] - dist - bottom_leg_len - bottom_len - precise_width),
                                     (base_width/2, treshhold)), 'or', squid_layer)


        # рисование вырезов на граунд плейне

        self.boolean(gdspy.Rectangle((-gp_cut_width / 2, 0),
                                     (gp_cut_width / 2, -gp_cut_heidth)).translate(dx=0,
                                                                                     dy=treshhold), 'or', 0)

        self.boolean(gdspy.Rectangle((-gp_cut_width - (interpad_dist + top_pad_width - gp_cut_width) / 2, h),
                                     (-interpad_dist / 2 - (top_pad_width - gp_cut_width) / 2,
                                      h + gp_cut_heidth)).translate(dx=0, dy=-treshhold), 'or', 0)

        self.boolean(gdspy.Rectangle((-gp_cut_width - (interpad_dist + top_pad_width - gp_cut_width) / 2, h),
                                     (-interpad_dist / 2 - (top_pad_width - gp_cut_width) / 2,
                                      h + gp_cut_heidth)).translate(dx=interpad_dist + top_pad_width,
                                                                     dy=-treshhold), 'or', 0)

        # область для литографа
        self.boolean(gdspy.Rectangle((-300/2, -300/2), (300/2, 300/2)), 'or', 5)


