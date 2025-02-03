import numpy as np
import gdspy
from matplotlib import pyplot as plt
from collections.abc import Iterable
import copy
from EDPy.settings import *
from EDPy.validator import *
from EDPy.functions import *
from EDPy.example import *

class AutoPath(Path, Validator):

    __slots__ = ('__max_radius',)

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
            return path, max_radius
            # return path, rad_vectors, max_radius

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
                return path, max_radius
                # return path, rad_vectors, max_radius
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
                return path, max_radius
                # return path, rad_vectors, max_radius

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
            return path, max_radius
            # return path, rad_vectors, max_radius