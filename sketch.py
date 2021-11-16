import gdspy
from element import *

class Sketch(Geometry):
    def __init__(self, name, unit=1e-6, precision=1e-9, layers=None):
        super().__init__()

        self.unit = unit
        self.precision = precision

        self.name = name
        self.lib = gdspy.GdsLibrary(name='library', unit=self.unit, precision=self.precision)
        self.cell = self.lib.new_cell(self.name)

        baseGeometry = gdspy.Round((0, 0), 100)
        for i in [0, 1, 2]:
            self.boolean(baseGeometry, 'or', i)

        del self.isInverted

    def setUnit(self, value):
        self.unit = value

    def setPrecision(self, value):
        self.precision = value

    def addGeometry(self, geometry):
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

    def draw(self, show=True):
        gdspy.current_library = self.lib
        self.cell.add(self.elements)
        if show:
            gdspy.LayoutViewer()

    def saveGDS(self, filename):
        self.lib.write_gds(filename + '.gds')

    def saveSVG(self, filename):
        self.cell.write_svg(filename + '.svg')
