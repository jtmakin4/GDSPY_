from EDTpy.settings import *
from collections.abc import Iterable
from EDTpy.validator import *
import copy
import gdspy
"""
Здесь находятся улучшенные версии функций GDSPY:

unique_layers и unique_datatypes - нахождение набора заполненных слоев объекта

extact - извлечение всех полигонов объекта из заданных слоёв и дататипов. в результате возвращается PolygonSet, 
исходный объект можнооставить без изменений или убрать извлеченне слои

append_geometry - добавление к PolygonSet-like объекту полигонов другого объекта с учетом слоёв и дататайпов (обратный extract)
не работает как булева операция, а только добавляет полигоны, что дает ускорение работы

boolean - быстрые булевы операции с учетом слоев и дататипов (в отличае от GDSPY варианта,
где результат сливается в один слой)
работает только с геометрией как таковой, не учитывает инвертированность слоёв и не работает с портами

"""

__all__ = ['unique_layers', 'unique_datatypes', 'extract', 'boolean', 'append_geometry']


def unique_layers(objs):
    if not isinstance(objs, Iterable):
        objs = [objs]
    else:
        objs = list(objs)

    layers = []
    for geom in objs:
        layers += list(set(geom.layers))
    return list(set(layers))


def unique_datatypes(objs):
    if not isinstance(objs, Iterable):
        objs = [objs]
    else:
        objs = list(objs)

    datatypes = []
    for geom in objs:
        datatypes += list(set(geom.datatypes))
    return list(set(datatypes))


def extract(base_obj, layers, datatypes=0, keep_obj=False):
    """
    :param base_obj: Объект PolygonSet, NewGeometry из которого извлекаются все полигоны определенного слоя и типа данных
    :param layers: номер извлекаемого слоя
    :param datatypes: тип данных извлекаемого слоя
    :param keep_obj: сохранить ли содержимое исходного объекта
    :return: PolygonSet, содержащий извлеченные данные. Если данные отсутствуют, возвращает None
    """
    Validator.polygone_geometry(base_obj)
    # я хз почему, но иногда в gdspy слои обозначаются не int, а numpy.int32
    if isinstance(layers, np.int32):
        layers = int(layers)
    if isinstance(datatypes, np.int32):
        datatypes = int(datatypes)

    # проверочки и коррекция типов
    if isinstance(layers, int):
        layers = [layers]
    elif isinstance(layers, list):
        pass
    else:
        raise ValueError('layers should be list or int')

    if isinstance(datatypes, int):
        datatypes = [datatypes]
    elif isinstance(datatypes, list):
        pass
    else:
        raise ValueError('datatypes should be list or int')

    # создаем копии сохраняемых и извлекаемых данных
    extracted_polygones, extracted_layers, extracted_datatypes = [], [], []
    saved_polygones, saved_layers, saved_datatypes = [], [], []
    # print(layers, base_obj.layers)
    for i in range(len(base_obj.polygons)):
        if (base_obj.layers[i] in layers) and (base_obj.datatypes[i] in datatypes):
            # print('extracted')
            extracted_polygones += [base_obj.polygons[i]]
            extracted_layers += [base_obj.layers[i]]
            extracted_datatypes += [base_obj.datatypes[i]]
        else:
            saved_polygones += [base_obj.polygons[i]]
            saved_layers += [base_obj.layers[i]]
            saved_datatypes += [base_obj.datatypes[i]]

    # запихиваем все что извлекли в результат
    result = gdspy.PolygonSet([])
    result.polygons = extracted_polygones
    result.layers = extracted_layers
    result.datatypes = extracted_datatypes

    # если не стоит флага, изменяем также исходный объект
    if keep_obj:
        pass
    else:
        base_obj.polygons = saved_polygones
        base_obj.layers = saved_layers
        base_obj.datatypes = saved_datatypes

    return result


def append_geometry(base_obj, tool_obj, keep_obj=False):
    """
    Функция для быстрого добавления полигонов одних объектов к другим, без их слияния
     (может быть особенно полезна для рисования via holes)
    :param base_obj: Объект, к которому добавляются полигоны
    :param tool_obj: Объект, содержащий добавляемые полигоны
    :param keep_obj: Флаг сохранения base_obj.
    Если не сохраняет, возвращает измененную копию объекта, остваляя исходный объект без изменений
    :returns: None or PolygonSet-like object
    """
    Validator.polygone_geometry(base_obj)
    Validator.polygone_geometry(tool_obj)
    if keep_obj:
        result = copy.deepcopy(base_obj)
        result.polygons += tool_obj.polygons
        result.layers += tool_obj.layers
        result.datatypes += tool_obj.datatypes
        return result
    else:
        base_obj.polygons += tool_obj.polygons
        base_obj.layers += tool_obj.layers
        base_obj.datatypes += tool_obj.datatypes
        return base_obj


def boolean(base_obj, tool_obj, operation, keep_obj=False, precision=Settings.PRECISION/Settings.UNIT, max_points=Settings.MAX_POLYGON_POINTS):

    Validator.polygone_geometry(base_obj)
    Validator.polygone_geometry(tool_obj)

    layers = list(set(unique_layers(base_obj) + unique_layers(tool_obj)))
    datatypes = list(set(unique_datatypes(base_obj) + unique_datatypes(tool_obj)))

    if keep_obj:
        result_obj = copy.deepcopy(base_obj)
    else:
        result_obj = base_obj

    for j in datatypes:
        for i in layers:
            base = extract(result_obj, i, j, keep_obj=False)
            tool = extract(tool_obj, i, j, keep_obj=True)
            result = gdspy.boolean(base, tool, operation, layer=i, datatype=j, precision=precision, max_points=max_points)
            if result is not None:
                append_geometry(result_obj, result, keep_obj=False)
    return result_obj


