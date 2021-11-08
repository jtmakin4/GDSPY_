import gdspy
import numpy as np
from CPW_calculator import *
from Parameters2 import *

curve_tolerance = 5e0
curve_precision = 5e-1

# tolerance >= precision


def layer_clear(cell, layer_n):
    cell.remove_polygons(lambda pts, layer, datatype: layer == layer_n)


def cell_clear(cell):
    cell.remove_polygons(lambda pts, layer, datatype: True)


def layer_merge(cell, layer):
    working = cell.get_polygons(by_spec=True)[(layer, 0)]
    layer_clear(cell, layer)
    result = gdspy.boolean(working, None, 'or', max_points=0, layer=layer)
    cell.add(result)


def layer_boolean(cell, layer, polygon, operation):
    working_polygons = cell.get_polygons(by_spec=True)[(layer, 0)]

    if len(working_polygons) != 0:
        result = gdspy.Polygon(working_polygons[0], layer=layer)
        for i in range(1, len(working_polygons)):
            mass = gdspy.Polygon(working_polygons[i], layer=layer)
            result = gdspy.boolean(result, mass, operation='or', layer=layer)
            del mass

        layer_clear(cell, layer)
        res = gdspy.boolean(result, polygon, operation=operation, layer=layer)
        res = gdspy.boolean(res, None, 'or', max_points=0, layer=layer)
        cell.add(res)
        del working_polygons, res


class CoplanarWaveguide:
    def __init__(self, s, w, g, curve_r, path):
        """
             Coplanar waveguide object. Objected to drawing waveguide going throught set path.

            :param S: Number.
                Width of central line.
            :param W: Number.
                Gap between central line and ground plane.
            :param path: Array-like[N][2].
                Points along the center of the path.
            :param curve_r: Number.
                Curvature radius.
        """
        self._s = s
        self._w = w
        self._g = g
        self._path = path
        self._r = curve_r
        self._in_point = path[0]
        self._out_point = path[-1]

    def _unit_vector(self, point1, point2):
        vec = np.array(point2) - np.array(point1)
        vec = vec / np.linalg.norm(vec)
        return vec

    def _array_simplify(self, input_array):
        array = np.array(input_array[0])
        for i in range(1, len(input_array) - 1):
            vec1 = self._unit_vector(input_array[i - 1], input_array[i])
            vec2 = self._unit_vector(input_array[i], input_array[i + 1])
            if vec1.dot(vec2) != 1:
                array = np.vstack([array, input_array[i]])
        array = np.vstack([array, input_array[-1]])
        return array

    def _circular_bend_path(self, input_array, bend_r):
        array = np.array(input_array[0])
        arc_centers = np.array([0, 0])
        angles = np.array([])
        for i in range(1, len(input_array) - 1):
            vec1 = self._unit_vector(input_array[i - 1], input_array[i])
            vec2 = self._unit_vector(input_array[i], input_array[i + 1])
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
        array = self._array_simplify(array)
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

    def _rotate_around(self, point, center, angle, diraction):
        if diraction > 0:
            rotate_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                         [np.sin(angle), np.cos(angle)]])
        else:
            rotate_matrix = np.array([[np.cos(angle), np.sin(angle)],
                                         [-np.sin(angle), np.cos(angle)]])
        return np.dot(rotate_matrix, point - center) + center

    def _len_to_position(self, l, points, centers, angles, bend_r):
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
            result_point = self._rotate_around(points[section_n], centers[section_n // 2], point_angle, diraction)
            vec = np.array(result_point - centers[section_n // 2]) / np.linalg.norm(
                result_point - centers[section_n // 2])
            return result_point, vec

    def _create_vias_array(self, path, bend_radius, offset, step):
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

    def _draw_cpw_mask(self):
        w_polygon = gdspy.FlexPath(self._path,
                                   self._s + 2 * self._w,
                                   layer=0,
                                   corners="circular bend",
                                   bend_radius=self._r,
                                   tolerance=curve_tolerance,
                                   precision=curve_precision)
        s_polygon = gdspy.FlexPath(self._path,
                                   self._s,
                                   layer=0,
                                   corners="circular bend",
                                   bend_radius=self._r,
                                   tolerance=curve_tolerance,
                                   precision=curve_precision)

        tech_polygon = gdspy.FlexPath(self._path,
                               self._s + 2 * self._w + 2 * (300 + via_d),
                               layer=tech_layer,
                               corners="circular bend",
                               bend_radius=self._r,
                               tolerance=curve_tolerance,
                               precision=curve_precision)

        w_polygon = gdspy.boolean(w_polygon, s_polygon, operation='not', layer=0)
        return w_polygon, tech_polygon

    def draw(self, layer, cell):
        """
            Drawing function

            :param target_polygon:
            :param layer: Int number.
                Layer number for CPW waveguide.
            :param cell: gdspy.Cell().
                Cell for CPW waveguide.
        """
        cpw_mask, tech_mask = self._draw_cpw_mask()
        layer_boolean(cell, layer, cpw_mask, 'not')
        layer_boolean(cell, tech_layer, tech_mask, 'not')


        offset = via_d + 300+ self._s/2 + self._w
        step = 500
        point = self._create_vias_array(self._path, self._r, offset, step)
        for i in range(0, len(point)):
            arc = gdspy.Round(
                point[i],
                via_d,
                tolerance=curve_tolerance*0.1,
                layer=layers['via_holes']
            )
            cell.add(arc)
        # return cpw_mask

class SMP_mounting:
    def __init__(self, place, angle):
        """
            SMP_mounting place. Created for drawing and manipulating of it.

            :param place: [2]tuple.
                Point of mounting hole center.
            :param angle: Number.
                Angle of counterclockwise rotation in pi. Initially goes from up to down direction.
        """
        self.place = place
        self.angle = angle
        self._out_point = (self.place[0] + 5 * 1e3 / 2 * np.sin(self.angle),
                           self.place[1] - 5 * 1e3 / 2 * np.cos(self.angle))

    def _draw_smp_mask_func(self):
        # Main part
        smp_mounting_mask = gdspy.Round((0, 0),
                                        2 * 1e3 / 2,
                                        tolerance=1e0)
        cpw_smp = gdspy.Rectangle((-1 * 1e3, 0),
                               (1e3, -5 * 1e3 / 2))
        smp_mounting_mask = gdspy.boolean(smp_mounting_mask, cpw_smp, 'or')
        cpw_smp = gdspy.Rectangle((-0.363 / 2 * 1e3, -5 * 1e3 / 2),  # 0.436
                               (0.363 / 2 * 1e3, -5 * 1e3 / 2 + 0.4 * 1e3))
        smp_mounting_mask = gdspy.boolean(smp_mounting_mask, cpw_smp, 'not')

        # Corners drawing
            # Rectangles drawing
        corner1 = gdspy.Rectangle((-5 * 1e3 / 2, 5 * 1e3 / 2 - 1.5 * 1e3),
                               (-5 * 1e3 / 2 + 0.363 * 1e3, 5 * 1e3 / 2))
        corner2 = gdspy.Rectangle((-5 * 1e3 / 2 + 0.363 * 1e3, 5 * 1e3 / 2),
                                (-5 * 1e3 / 2 + 1.5 * 1e3, 5 * 1e3 / 2 - 0.363 * 1e3))
            # First corner drawing
        corner1 = gdspy.boolean(corner1, corner2, 'or')
        corner2 = gdspy.copy(corner1)
            # Second corner drawing
        corner2.mirror((0, 1e3), (0, 0))

            # tech mask drawing
        tech_mask = gdspy.Rectangle((-5 * 1e3 / 2 - (tech_dist+via_d),
                                 5 * 1e3 / 2 + (tech_dist+via_d)),
                               (5 * 1e3 / 2 + (tech_dist+via_d), -5 * 1e3 / 2 - (tech_dist+via_d)))
        tech_mask.rotate(self.angle)
        tech_mask.translate(self.place[0], self.place[1])

        # Result
        corner1 = gdspy.boolean(corner1, corner2, 'or')
        smp_mounting_mask = gdspy.boolean(smp_mounting_mask, corner1, 'or')
        del(corner1, corner2, cpw_smp)
        return smp_mounting_mask, tech_mask

    def draw(self, layer, cell):
        """
            Drawing function

            :param target_polygon:
            :param layer: Int number.
                Layer number for CPW waveguide.
            :param cell: gdspy.Cell().
                Cell for CPW waveguide.
        """

        smp_mounting_mask, tech_smp_mounting_mask = self._draw_smp_mask_func()

        smp_mounting_mask.rotate(self.angle)
        smp_mounting_mask.translate(self.place[0], self.place[1])

        layer_boolean(cell, layer, smp_mounting_mask, 'not')
        layer_boolean(cell, tech_layer, tech_smp_mounting_mask, 'not')


    def coupling_place(self):
        """
        Calculates place for begining of CPW.

        :return: [2]tuple point.
        """
        return self._out_point

    def _draw_cpw_connector_func(self, s, w, length):
        w_polygon = gdspy.Path(2 * 1e3,
                               (0, -5 / 2 * 1e3),
                               number_of_paths=1)
        w_polygon.parametric(curve_function=lambda u: (0, -u * (length - 1e3)),
                             final_width=lambda u: ((2 - 0.363) / 2 * 1e3 + w + (0.363 * 1e3 + s) / 2)
                                                   + ((2 - 0.363) / 2 * 1e3 - w + (0.363 * 1e3 - s) / 2)
                                                   * np.cos(np.pi * u),
                             tolerance=0.1,
                             number_of_evaluations=50)

        s_polygon = gdspy.Path(2 * 1e3,
                               (0, -5 / 2 * 1e3),
                               number_of_paths=1)
        s_polygon.parametric(curve_function=lambda u: (0, -u * (length - 1e3)),
                             final_width=lambda u: (0.363 * 1e3 + s) / 2
                                                   + (0.363 * 1e3 - s) / 2 * np.cos(np.pi * u),
                             tolerance=0.1,
                             number_of_evaluations=50)

        tech_poly = gdspy.Path(2 * 1e3 + (300 + via_d) * 2,
                           (0, -5 / 2 * 1e3),
                           number_of_paths=1)

        tech_poly.parametric(curve_function=lambda u: (0, -u*(length - 1e3)),
                         final_width=lambda u: ((2 - 0.363)/2*1e3 + w + (0.363*1e3 + s)/2)
                                                + ((2 - 0.363)/2*1e3 - w
                                                + (0.363*1e3 - s)/2)*np.cos(np.pi*u)+(300+via_d)*2,
                         tolerance=0.1,
                         number_of_evaluations=50)

        # (0,0) coordinate corresponds to the last coordinate of previous segment. u varies from 0 to 1.

        w_polygon = gdspy.boolean(w_polygon, s_polygon, 'not')
        return w_polygon, tech_poly

    def draw_cpw_connector(self, s, w, length, layer, cell):
        """
        Draws connector between mounting hole and CPW. Supposed that CPW is 50 Om.

        :param w:
        :param s:
        :param target_polygon:
        :param S: Number.
            Width of central line connected with mounting hole.
        :param W: Number.
            Gap between central line and ground plane connected with mounting hole.
        :param length: Number.
            Length of connector. Minimum is 1e3.
        :param layer: Int number.
                Layer number for CPW connector .
        :param cell: gdspy.Cell().
                Cell for CPW connector.
        """
        self._out_point = (self.place[0] + (5/2*1e3 + length-1e3) * np.sin(self.angle),
                           self.place[1] - (5/2*1e3 + length-1e3) * np.cos(self.angle))
        if length < 1e3:
            print('enter length more than 1e3')
            return None
        connector_mask, tech_connector_mask = self._draw_cpw_connector_func(s, w, length)

        connector_mask.rotate(self.angle)
        connector_mask.translate(self.place[0], self.place[1])
        tech_connector_mask.rotate(self.angle)
        tech_connector_mask.translate(self.place[0], self.place[1])
        layer_boolean(cell, layer, connector_mask, 'not')
        layer_boolean(cell, tech_layer, tech_connector_mask, 'not')


# отверстия ставятся после формирования слоев платы!!!!!!!
class Hole:
    def __init__(self, shape, params, metallised):
        self.shape = shape
        self.metallised = metallised
        self.params = params
        self.tech_dist = 300
        self.via_d = 1000/2 # radius of creating holes

    def draw(self, place, angle, metal_layers, substrate_layer, cell):
        if self.shape == "rectangle":
            elem = gdspy.Rectangle((-self.params[0]/2, -self.params[1]/2),
                                   (self.params[0]/2, self.params[1]/2))

            holes = gdspy.Round((-self.params[0]/2 + self.via_d, -self.params[1]/2),  # smp is main object
                                self.via_d,
                                tolerance=curve_tolerance)
            holes = gdspy.boolean(holes, gdspy.Round((-self.params[0] / 2 + self.via_d, self.params[1] / 2),
                                                     self.via_d,
                                                     tolerance=curve_tolerance), 'or')
            holes = gdspy.boolean(holes, gdspy.Round((self.params[0] / 2 - self.via_d, -self.params[1] / 2),
                                                     self.via_d,
                                                     tolerance=curve_tolerance), 'or')
            holes = gdspy.boolean(holes, gdspy.Round((self.params[0] / 2 - self.via_d, self.params[1] / 2),
                                                     self.via_d,
                                                     tolerance=curve_tolerance), 'or')
            elem = gdspy.boolean(elem, holes, 'or')
            elem.rotate(angle)
            elem.translate(place[0], place[1])
            layer_boolean(cell, substrate_layer, elem, 'not')

            # tech layer drawing
            # todo сделать человеческие названия

            elem_dist = gdspy.Rectangle((-self.params[0] / 2 - via_d - tech_dist - self.tech_dist,
                                         -self.params[1]/2 - via_d - tech_dist - self.tech_dist),
                                        (self.params[0] / 2 + via_d + tech_dist + self.tech_dist,
                                         self.params[1] / 2 + via_d + tech_dist + self.tech_dist))

            holes_dist = gdspy.Round((-self.params[0] / 2 + self.via_d, -self.params[1] / 2),  # smp is main object
                                     self.via_d + via_d + tech_dist + self.tech_dist,
                                     tolerance=curve_tolerance)

            holes_dist = gdspy.boolean(holes_dist, gdspy.Round((-self.params[0] / 2 + self.via_d, self.params[1] / 2),
                                                               self.via_d + via_d + tech_dist + self.tech_dist,
                                                               tolerance=curve_tolerance), 'or')

            holes_dist = gdspy.boolean(holes_dist, gdspy.Round((self.params[0] / 2 - self.via_d, -self.params[1] / 2),
                                                               self.via_d + via_d + tech_dist + self.tech_dist,
                                                               tolerance=curve_tolerance), 'or')

            holes_dist = gdspy.boolean(holes_dist, gdspy.Round((self.params[0] / 2 - self.via_d, self.params[1] / 2),
                                                               self.via_d + via_d + tech_dist + self.tech_dist,
                                                               tolerance=curve_tolerance), 'or')

            elem_dist = gdspy.boolean(elem_dist, holes_dist, 'or')
            elem_dist.rotate(angle)
            elem_dist.translate(place[0], place[1])
            layer_boolean(cell,tech_layer, elem_dist, 'not')

            if self.metallised:
                for i in metal_layers:
                    layer_boolean(cell, i, elem, 'not')
            else:
                elem = gdspy.Rectangle((-self.params[0] / 2 - self.tech_dist, -self.params[1] / 2 - self.tech_dist),
                                       (self.params[0] / 2 + self.tech_dist, self.params[1] / 2 + self.tech_dist))

                holes = gdspy.Round((-self.params[0] / 2 + self.via_d, -self.params[1] / 2),  # smp is main object
                                    self.via_d + self.tech_dist,
                                    tolerance=curve_tolerance)

                holes = gdspy.boolean(holes, gdspy.Round((-self.params[0] / 2 + self.via_d, self.params[1] / 2),
                                                         self.via_d + self.tech_dist,
                                                         tolerance=curve_tolerance), 'or')

                holes = gdspy.boolean(holes, gdspy.Round((self.params[0] / 2 - self.via_d, -self.params[1] / 2),
                                                         self.via_d + self.tech_dist,
                                                         tolerance=curve_tolerance), 'or')

                holes = gdspy.boolean(holes, gdspy.Round((self.params[0] / 2 - self.via_d, self.params[1] / 2),
                                                         self.via_d + self.tech_dist,
                                                         tolerance=curve_tolerance), 'or')

                elem = gdspy.boolean(elem, holes, 'or')

                elem.rotate(angle)
                elem.translate(place[0], place[1])

                for i in metal_layers:
                    layer_boolean(cell, i, elem, 'not')

        if self.shape == "circle":
            elem = gdspy.Round((0, 0),
                               self.params[0]/2,
                               tolerance=curve_tolerance)

            elem_dist = gdspy.Round((0, 0),
                                    self.params[0] / 2 + self.tech_dist + tech_dist,
                                    tolerance=curve_tolerance)

            elem.translate(place[0], place[1])
            elem_dist.translate(place[0], place[1])
            layer_boolean(cell, substrate_layer, elem, 'not')
            layer_boolean(cell,tech_layer, elem_dist, 'not')

            if self.metallised:
                for i in metal_layers:
                    layer_boolean(cell, i, elem, 'not')
            else:
                elem = gdspy.Round((0, 0),
                                   self.params[0]/2 + self.tech_dist,
                                   tolerance=curve_tolerance)

                elem.translate(place[0], place[1])
                for i in metal_layers:
                    layer_boolean(cell, i, elem, 'not')



# # плохие функции
# def create_lattice(point, a_vec, b_vec, cell):
#     working = cell.get_polygons(by_spec=True)
#     keys = list(working.keys())
#
#     x_num = 50
#     y_num = x_num
#
#     result = []
#     for i in range(-x_num,x_num):
#         for j in range(-y_num,y_num):
#             p1 = [(point[0] + i * a_vec[0] + j * b_vec[0], point[1] + i * a_vec[1] + j * b_vec[1])]
#             p2 = [(point[0] - i * a_vec[0] + j * b_vec[0], point[1] - i * a_vec[1] + j * b_vec[1])]
#             p3 = [(point[0] + i * a_vec[0] - j * b_vec[0], point[1] + i * a_vec[1] - j * b_vec[1])]
#             p4 = [(point[0] - i * a_vec[0] - j * b_vec[0], point[1] - i * a_vec[1] - j * b_vec[1])]
#             # flag = []
#             # for k in keys:
#             #     flag.append(all(gdspy.inside(p, working[k])))
#             # if all(flag):
#             #     result.append(p[0])
#             if all(gdspy.inside(p1, working[tech_layer,0)])):
#                 result.append(p1[0])
#             if all(gdspy.inside(p2, working[tech_layer, 0)])):
#                     result.append(p2[0])
#             if all(gdspy.inside(p3, working[tech_layer, 0)])):
#                 result.append(p3[0])
#             if all(gdspy.inside(p4, working[tech_layer, 0)])):
#                     result.append(p4[0])
#
#     return result
#
#
# def draw_vias(d, a_vec, b_vec, layer, cell):
#     lattice = create_lattice((0, 0), a_vec, b_vec, cell)
#     elem = gdspy.Round(lattice[0],
#                        d / 2,
#                        tolerance=1e0,
#                        layer = layer)
#     for i, center in enumerate(lattice[0:]):
#         elem2 = gdspy.Round(center,
#                             d / 2,
#                             tolerance=1e0,
#                             layer = layer)
#         elem = gdspy.boolean(elem, elem2, 'or', layer = layer)
#
#     cell.add(elem)
#     del(elem)
