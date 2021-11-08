import gdspy
import numpy as np
from CPW_calculator import *
from Parameters import *

curve_tolerance = 5e0
curve_precision = 5e-1
#tolerance >= precision

def layer_clear(cell, layer_N):
    """
    Function for delining all poligons belonging to layer_N.

    :param cell: gdspy.Cell.
        The cell that the function is applied to.
    :param layer_N: Int number.
        The layer for deleting.
    """
    cell.remove_polygons(lambda pts, layer, datatype: layer == layer_N)

def cell_clear(cell):
        cell.remove_polygons(lambda pts, layer, datatype: True)

def layer_boolean(cell, layer, polygon, operation):
    """
    Boolean function applied to layers.

    :param cell: gdspy.Cell.
        The cell that the function is applied to.
    :param layer: Int number.
        The layer for boolean operation. First operand.
    :param polygon: gdspy.Polygon.
        The polygon for boolean operation. Second operand.
    :param operation: ({'or', 'and', 'xor', 'not'})
    :return: If wrong operation, than None.
    """
    working = cell.get_polygons(by_spec = True)
    keys = list(working.keys())
    flag = False
    for i in keys:
        flag = (i == (layer,0)) or flag
    if flag:
        working = working[(layer, 0)]
        result = gdspy.Polygon(working[0], layer = layer)
        for i in range(1,len(working)):
            mass = gdspy.Polygon(working[i], layer = layer)
            result = gdspy.boolean(result, mass, operation = 'or', layer=layer)
            del(mass)

        # layer_clear(cell, layer)
        # res = gdspy.boolean(result, polygon, operation = operation, layer = layer)

        layer_clear(cell, layer)
        res = gdspy.boolean(result, polygon, operation = operation, layer = layer)
        res = gdspy.boolean(res, None, 'or', max_points=0, layer=layer)
        cell.add(res)
        # if res != None:
        #     cell.add(res)
        del(working)
        del(res)





class CPW():

    def __init__(self, S, W, path, curve_r):
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
        self._S = S
        self._W = W
        self._path = path
        self._R = curve_r

    def draw(self, layer, cell):
        """
            Drawing function

            :param layer: Int number.
                Layer number for CPW waveguide.
            :param cell: gdspy.Cell().
                Cell for CPW waveguide.
        """
        path1 = gdspy.FlexPath(self._path,
                               self._S + 2 * self._W,
                               layer = layer,
                               corners = "circular bend",
                               bend_radius = self._R,
                               tolerance = curve_tolerance,
                               precision = curve_precision)
        path2 = gdspy.FlexPath(self._path,
                               self._S,
                               layer = layer,
                               corners = "circular bend",
                               bend_radius = self._R,
                               tolerance = curve_tolerance,
                               precision = curve_precision)
        path3 = gdspy.FlexPath(self._path,
                               self._S + 2 * self._W + 2*(300+drill_r),
                               layer = layer,
                               corners = "circular bend",
                               bend_radius = self._R,
                               tolerance = curve_tolerance,
                               precision = curve_precision)
        #todo ускорить функцию - layer boolean медленная
        path1 = gdspy.boolean(path1,path2,'not')
        layer_boolean(cell, layer, path1, 'not')
        # layer_boolean(cell, layer, path2, 'or')
        layer_boolean(cell, 24, path3, 'not') # dist_layer

    def draw_mask(self, layer, cell):
        path1 = gdspy.FlexPath(self._path,
                               self._S + 2 * self._W,
                               layer=layer,
                               corners="circular bend",
                               bend_radius=self._R,
                               tolerance=curve_tolerance,
                               precision=curve_precision)
        path2 = gdspy.FlexPath(self._path,
                               self._S,
                               layer=layer,
                               corners="circular bend",
                               bend_radius=self._R,
                               tolerance=curve_tolerance,
                               precision=curve_precision)
        path3 = gdspy.FlexPath(self._path,
                               self._S + 2 * self._W + 2 * (300 + drill_r),
                               layer=layer,
                               corners="circular bend",
                               bend_radius=self._R,
                               tolerance=curve_tolerance,
                               precision=curve_precision)
        # todo ускорить функцию - layer boolean медленная
        path1 = gdspy.boolean(path1, path2, 'not')
        return(path1, path3)



class SMP_mounting():
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
        self.CPW_coordinates = (self.place[0] + 5 * 1e3 / 2 * np.sin(self.angle),
                                self.place[1] - 5 * 1e3 / 2 * np.cos(self.angle))



    def draw(self, layer, cell):
        """
            Drawing function

            :param layer: Int number.
                Layer number for CPW waveguide.
            :param cell: gdspy.Cell().
                Cell for CPW waveguide.
        """
        # main draw part
        # smp = gdspy.Round((0, 0),  # smp is main object
        #                   4 * 1e3 / 2,
        #                   tolerance=1e0)

        smp = gdspy.Round((0, 0),  # smp is main object
                          2 * 1e3 / 2,
                          tolerance=1e0)

        elem = gdspy.Rectangle((-1 * 1e3, 0),
                               (1e3, -5 * 1e3 / 2))
        smp = gdspy.boolean(smp, elem, 'or')
        elem = gdspy.Rectangle((-0.363/2 * 1e3, -5 * 1e3 / 2),# 0.436
                               (0.363/2 * 1e3, -5 * 1e3 / 2 + 0.4 * 1e3))
        smp = gdspy.boolean(smp, elem, 'not')

        # corners drawing
        elem = gdspy.Rectangle((-5 * 1e3 / 2, 5 * 1e3 / 2 - 1.5 * 1e3),
                               (-5 * 1e3 / 2 + 0.363 * 1e3, 5 * 1e3 / 2))
        elem2 = gdspy.Rectangle((-5 * 1e3 / 2 + 0.363 * 1e3, 5 * 1e3 / 2),
                                (-5 * 1e3 / 2 + 1.5 * 1e3, 5 * 1e3 / 2 - 0.363 * 1e3))
        elem = gdspy.boolean(elem, elem2, 'or')
        elem2 = gdspy.copy(elem)
        elem2.mirror((0, 1e3), (0, 0))
        elem = gdspy.boolean(elem, elem2, 'or')
        smp = gdspy.boolean(smp, elem, 'or')

        elem3 = gdspy.Rectangle((-5 * 1e3 / 2 - (tech_dist+drill_r),
                                 5 * 1e3 / 2 + (tech_dist+drill_r)),
                               (5 * 1e3 / 2 + (tech_dist+drill_r), -5 * 1e3 / 2 - (tech_dist+drill_r)))
        # rotation and translation part
        smp.rotate(self.angle)
        smp.translate(self.place[0], self.place[1])
        elem3.rotate(self.angle)
        elem3.translate(self.place[0], self.place[1])

        # creation part
        layer_boolean(cell, layer, smp, 'not')
        layer_boolean(cell, 24, elem3, 'not') #dist_layer
        # self.CPW_coordinates = self.CPW_place()

    def CPW_place(self):
        """
        Calculates place for begining of CPW.

        :return: [2]tuple point.
        """
        return self.CPW_coordinates

    def draw_CPW_connector(self, S, W, length, layer, cell):
        """
        Draws connector between mounting hole and CPW. Supposed that CPW is 50 Om.

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

        # торчащий пиптик имеет практически в точности 50 Ом!!!!
        if length < 1e3:
            print('enter length more than 1e3')
            return None

        # drawing part
        connect = gdspy.Path(2*1e3,
                             (0, -5/2*1e3),
                             number_of_paths=1)
        elem = gdspy.Path(2*1e3,
                          (0, -5/2*1e3),
                          number_of_paths=1)
        elem3 = gdspy.Path(2*1e3+(300+drill_r)*2,
                          (0, -5/2*1e3),
                          number_of_paths=1)
        # connect.segment(1e3, "-y")
        # elem.segment(1e3, "-y")
        # elem3.segment(1e3, "-y")

        connect.parametric(curve_function = lambda u: (0, - u*(length - 1e3)),
                           final_width = lambda u: ((2 - 0.363)/2*1e3 + W + (0.363*1e3 + S)/2)
                                                   + ((2 - 0.363)/2*1e3 - W + (0.363*1e3 - S)/2)*np.cos(np.pi*u),
                           tolerance = 0.1,
                           number_of_evaluations = 50)

        elem.parametric(curve_function = lambda u: (0, - u*(length - 1e3)),
                        final_width = lambda u: (0.363*1e3 + S)/2
                                                   + (0.363*1e3 - S)/2*np.cos(np.pi*u),
                        tolerance = 0.1,
                        number_of_evaluations = 50)


        elem3.parametric(curve_function = lambda u: (0, - u*(length - 1e3)),
                           final_width = lambda u: ((2 - 0.363)/2*1e3 + W + (0.363*1e3 + S)/2)
                                                   + ((2 - 0.363)/2*1e3 - W +
                                                      (0.363*1e3 - S)/2)*np.cos(np.pi*u)+(300+drill_r)*2,
                           tolerance = 0.1,
                           number_of_evaluations = 50)

        # (0,0) coordinate corresponds to the last coordinate of previous segment. u varies from 0 to 1.
        connect = gdspy.boolean(connect, elem, 'not')

        # rotation and translation part
        connect.rotate(self.angle)
        connect.translate(self.place[0], self.place[1])
        elem3.rotate(self.angle)
        elem3.translate(self.place[0], self.place[1])

        # creation part
        layer_boolean(cell, layer, connect, 'not')
        layer_boolean(cell, 24, elem3, 'not') #dist_layer

        self.CPW_coordinates = (self.place[0] + (5/2*1e3 + length-1e3) * np.sin(self.angle),
                                self.place[1] - (5/2*1e3 + length-1e3) * np.cos(self.angle))

class Hole():
    def __init__(self, shape, params, metallised):
        self.shape = shape
        self.metallised = metallised
        self.params = params
        self.tech_dist = 300
        self.drill_r = 1000/2 # rardius of creating holes

    def draw(self, place, angle, metal_layers, substr_layer, cell):
        if (self.shape == "rectangle" ):
            elem = gdspy.Rectangle((-self.params[0]/2, -self.params[1]/2),
                                   (self.params[0]/2, self.params[1]/2))

            holes = gdspy.Round((-self.params[0]/2 + self.drill_r, -self.params[1]/2),  # smp is main object
                                self.drill_r,
                                tolerance = curve_tolerance)
            holes = gdspy.boolean(holes, gdspy.Round((-self.params[0] / 2 + self.drill_r, self.params[1] / 2),
                                                     self.drill_r,
                                                     tolerance = curve_tolerance), 'or')
            holes = gdspy.boolean(holes, gdspy.Round((self.params[0] / 2 - self.drill_r, -self.params[1] / 2),
                                                     self.drill_r,
                                                     tolerance = curve_tolerance), 'or')
            holes = gdspy.boolean(holes, gdspy.Round((self.params[0] / 2 - self.drill_r, self.params[1] / 2),
                                                     self.drill_r,
                                                     tolerance = curve_tolerance), 'or')
            elem = gdspy.boolean(elem, holes, 'or')
            elem.rotate(angle)
            elem.translate(place[0], place[1])
            layer_boolean(cell, substr_layer, elem, 'not')
            # dist layer drawing

            elem_dist = gdspy.Rectangle((-self.params[0] / 2 - drill_r - tech_dist - self.tech_dist,
                                         -self.params[1]/2 - drill_r - tech_dist - self.tech_dist),
                                   (self.params[0] / 2 + drill_r + tech_dist + self.tech_dist,
                                    self.params[1] / 2  + drill_r + tech_dist + self.tech_dist))

            holes_dist = gdspy.Round((-self.params[0] / 2 + self.drill_r, -self.params[1] / 2),  # smp is main object
                                self.drill_r + drill_r + tech_dist + self.tech_dist,
                                tolerance=curve_tolerance)
            holes_dist = gdspy.boolean(holes_dist, gdspy.Round((-self.params[0] / 2 + self.drill_r, self.params[1] / 2),
                                                               self.drill_r + drill_r + tech_dist + self.tech_dist,
                                                     tolerance=curve_tolerance), 'or')
            holes_dist = gdspy.boolean(holes_dist, gdspy.Round((self.params[0] / 2 - self.drill_r, -self.params[1] / 2),
                                                     self.drill_r + drill_r + tech_dist + self.tech_dist,
                                                     tolerance=curve_tolerance), 'or')
            holes_dist = gdspy.boolean(holes_dist, gdspy.Round((self.params[0] / 2 - self.drill_r, self.params[1] / 2),
                                                     self.drill_r + drill_r + tech_dist + self.tech_dist,
                                                     tolerance=curve_tolerance), 'or')

            elem_dist = gdspy.boolean(elem_dist, holes_dist, 'or')
            elem_dist.rotate(angle)
            elem_dist.translate(place[0], place[1])
            layer_boolean(cell, 24, elem_dist, 'not') #dist layer





            if(self.metallised):
                for i in metal_layers:
                    layer_boolean(cell, i, elem, 'not')
            else:
                elem = gdspy.Rectangle((-self.params[0] / 2 - self.tech_dist, -self.params[1] / 2 - self.tech_dist),
                                       (self.params[0] / 2 + self.tech_dist, self.params[1] / 2 + self.tech_dist))

                holes = gdspy.Round((-self.params[0] / 2 + self.drill_r, -self.params[1] / 2),  # smp is main object
                                    self.drill_r + self.tech_dist,
                                    tolerance = curve_tolerance)
                holes = gdspy.boolean(holes, gdspy.Round((-self.params[0] / 2 + self.drill_r, self.params[1] / 2),
                                                         self.drill_r + self.tech_dist,
                                                         tolerance = curve_tolerance), 'or')
                holes = gdspy.boolean(holes, gdspy.Round((self.params[0] / 2 - self.drill_r, -self.params[1] / 2),
                                                         self.drill_r + self.tech_dist,
                                                         tolerance = curve_tolerance), 'or')
                holes = gdspy.boolean(holes, gdspy.Round((self.params[0] / 2 - self.drill_r, self.params[1] / 2),
                                                         self.drill_r + self.tech_dist,
                                                         tolerance = curve_tolerance), 'or')
                elem = gdspy.boolean(elem, holes, 'or')

                elem.rotate(angle)
                elem.translate(place[0], place[1])


                for i in metal_layers:
                    layer_boolean(cell, i, elem, 'not')

        if (self.shape == "circle" ):
                elem = gdspy.Round((0,0),
                                self.params[0]/2,
                                tolerance = curve_tolerance)
                elem_dist = gdspy.Round((0, 0),
                                   self.params[0] / 2 + self.tech_dist + tech_dist,
                                   tolerance=curve_tolerance)

                elem.translate(place[0], place[1])
                elem_dist.translate(place[0], place[1])
                layer_boolean(cell, substr_layer, elem, 'not')
                layer_boolean(cell, 24, elem_dist, 'not')

                if (self.metallised):
                    for i in metal_layers:
                        layer_boolean(cell, i, elem, 'not')
                else:
                    elem = gdspy.Round((0,0),
                                       self.params[0]/2 + self.tech_dist,
                                       tolerance=curve_tolerance)

                    elem.translate(place[0], place[1])
                    for i in metal_layers:
                        layer_boolean(cell, i, elem, 'not')

def create_lattice(point, a_vec, b_vec, cell):
    working = cell.get_polygons(by_spec=True)
    keys = list(working.keys())

    x_num = 50
    y_num = x_num

    result = []
    for i in range(-x_num,x_num):
        for j in range(-y_num,y_num):
            p1 = [(point[0] + i * a_vec[0] + j * b_vec[0], point[1] + i * a_vec[1] + j * b_vec[1])]
            p2 = [(point[0] - i * a_vec[0] + j * b_vec[0], point[1] - i * a_vec[1] + j * b_vec[1])]
            p3 = [(point[0] + i * a_vec[0] - j * b_vec[0], point[1] + i * a_vec[1] - j * b_vec[1])]
            p4 = [(point[0] - i * a_vec[0] - j * b_vec[0], point[1] - i * a_vec[1] - j * b_vec[1])]
            # flag = []
            # for k in keys:
            #     flag.append(all(gdspy.inside(p, working[k])))
            # if all(flag):
            #     result.append(p[0])
            if all(gdspy.inside(p1, working[(24,0)])):
                result.append(p1[0])
            if all(gdspy.inside(p2, working[(24, 0)])):
                    result.append(p2[0])
            if all(gdspy.inside(p3, working[(24, 0)])):
                result.append(p3[0])
            if all(gdspy.inside(p4, working[(24, 0)])):
                    result.append(p4[0])

    return result




# def draw_vias(d, a_vec, b_vec, metal_layers, substr_layer, cell):
#     lattice = create_lattice((-20e3, -20e3), a_vec, b_vec, cell)
#
#     elem = gdspy.Round(lattice[0],
#                        d / 2,
#                        tolerance=1e0)
#     for i, center in enumerate(lattice[0:]):
#         elem = gdspy.boolean(elem,
#                              gdspy.Round(center,
#                                         d / 2,
#                                         tolerance=1e0),
#                              'or')
#         print(i)
#
#     layer_boolean(cell, substr_layer, elem, 'not')
#     for i in metal_layers:
#         layer_boolean(cell, i, elem, 'not')
#
#     del(elem)

def draw_vias(d, a_vec, b_vec, layer, cell):
    #todo включить возможность ресования дырок как дырок(см закоментированную функцию сверху)
    # lattice = create_lattice((-25e3, -25e3), a_vec, b_vec, cell)
    lattice = create_lattice((0, 0), a_vec, b_vec, cell)




    elem = gdspy.Round(lattice[0],
                       d / 2,
                       tolerance=1e0,
                       layer = layer)

    for i, center in enumerate(lattice[0:]):
        elem2 = gdspy.Round(center,
                            d / 2,
                            tolerance=1e0,
                            layer = layer)
        elem = gdspy.boolean(elem, elem2, 'or', layer = layer)
        # print(i)
        # sys.stderr.write('%d\r' % i)
    cell.add(elem)

    # layer_boolean(cell, substr_layer, elem, 'not')
    # for i in metal_layers:
    #     layer_boolean(cell, i, elem, 'not')

    del(elem)

def merging(cell,layer):
    working = cell.get_polygons(by_spec=True)
    working = working[(layer, 0)]
    layer_clear(cell, layer)
    result = gdspy.boolean(working, None, 'or', max_points=0, layer=layer)
    cell.add(result)

# def layer_simplify(cell, layer):
#     working = cell.get_polygons(by_spec = True)
#     keys = list(working.keys())
#     flag = False
#     for i in keys:
#         flag = (i == (layer,0)) or flag
#     if flag:
#         working = working[(layer, 0)]
#         result = gdspy.Polygon(working[0], layer = layer)
#         for i in range(1,len(working)):
#             mass = gdspy.Polygon(working[i], layer = layer)
#             result = gdspy.boolean(result, mass, operation = 'or', layer=layer)
#             del(mass)
#
#         layer_clear(cell, layer)
#         cell.add(result)
#
#         del(working)
#         del(result)