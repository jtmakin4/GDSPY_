import gdspy
from element import *

class Sketch(Geometry):
    def __init__(self, name, unit=1e-6, precision=1e-9, layers=None):
        super().__init__()
        del self.isInverted
        del self.ports

        self.geometries = []

        self.unit = unit
        self.precision = precision

        self.name = name
        self.lib = gdspy.GdsLibrary(name='library', unit=self.unit, precision=self.precision)
        self.cell = self.lib.new_cell(self.name)

        # baseGeometry = gdspy.Round((0, 0), 53e3/2)
        # for i in [0, 1, 2]:
        #     self.boolean(baseGeometry, 'or', i)


    def setUnit(self, value):
        self.unit = value

    def setPrecision(self, value):
        self.precision = value

    def addGeometry(self, geometry):
        geometry.flaot = False
        self.geometries.append(geometry)
        for i in geometry.layers:
            if geometry.isInverted[i]:
                try:
                    self.boolean(geometry.elements[i], 'not', i)
                except IndexError:
                    pass
            else:
                try:
                    self.boolean(geometry.elements[i], 'or', i)
                except IndexError:
                    pass

    def draw(self, show=True, showPorts = False):
        gdspy.current_library = self.lib
        self.cell.add(self.elements)

        # рисование портов
        if showPorts:
            for geometry in self.geometries:
                for i in range(0, len(geometry.ports)):
                    label = gdspy.Label(str(i), geometry.ports[i].position, "nw", geometry.ports[i].angle, layer=self.portsLayer)
                    self.cell.add(label)
                    a, o = geometry.ports[i].basis()

                    port_path = gdspy.Polygon([np.array(geometry.ports[i].position) + 1e3 * a,
                                               np.array(geometry.ports[i].position) + 0.2e3 * o,
                                               np.array(geometry.ports[i].position) - 0.2e3 * o], layer=self.portsLayer)

                    self.cell.add(port_path)

        if show:
            gdspy.LayoutViewer(pattern={'default': 0})

    def clearLayer(self, layer_n):
        self.cell.remove_polygons(lambda pts, layer, datatype: layer == layer_n)

    def clearCell(self):
        self.cell.remove_polygons(lambda pts, layer, datatype: True)

    def refine(self):
        for layer in range(0, 255):
            try:
                working = self.cell.get_polygons(by_spec=True)[(layer, 0)]
                self.clearLayer(layer)
                result = gdspy.boolean(working, None, 'or', max_points=0, layer=layer)
                self.cell.add(result)
            except KeyError:
                pass


    def saveGDS(self, filename):
        self.lib.write_gds(filename + '.gds')

    def saveSVG(self, filename):
        self.cell.write_svg(filename + '.svg')

