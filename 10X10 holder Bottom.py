import gdspy
import numpy as np
from Elements import *
from CPW_calculator import *
from Parameters import *


layers = {
    'bottom_metal' : 0,
    'substrate' : 1,
    'top_metal' : 2,
    'technical_layer0': 13,
    'technical_layer1': 14,
    'technical_layer2' : 15,
    'via_holes' : 16,
    'dist_layer': 24
}
filename = '10X10 PCB bottom'

lib = gdspy.GdsLibrary(name = 'library',unit = unit, precision = precision) # importing gdsii library
cell = lib.new_cell(filename) # setting working cell


circle = gdspy.Round((0, 0),
                     main_r,
                     layer = layers['substrate'],
                     tolerance = 1e1)
circle_metal_bottom = gdspy.Round((0, 0),
                                  main_r,
                                  layer = layers['bottom_metal'],
                                  tolerance = 1e1)
circle_metal_top = gdspy.Round((0, 0),
                               main_r,
                               layer = layers['top_metal'],
                               tolerance = 1e1)
circle_dist = gdspy.Round((0, 0),
                     main_r - 300 - drill_r,
                     layer = layers['dist_layer'],
                     tolerance = 1e1)


cell.add(circle_metal_top)
cell.add(circle)
cell.add(circle_metal_bottom)
cell.add(circle_dist)

# circle = gdspy.Round((-main_r,0),
#                      4e3/2,
#                      layer = layers['substrate'],
#                      tolerance = 1e1)
# circle_metal_bottom = gdspy.Round((-main_r,0),
#                                   4e3/2,
#                                   layer = layers['bottom_metal'],
#                                   tolerance = 1e1)
# circle_metal_top = gdspy.Round((-main_r,0),
#                                4e3/2,
#                                layer = layers['top_metal'],
#                                tolerance = 1e1)
# # circle_dist = gdspy.Round((-main_r,0),
# #                      4e3/2 - 300 - drill_r,
# #                      layer = layers['dist_layer'],
# #                      tolerance = 1e1)
#
#
# cell.add(circle_metal_top)
# cell.add(circle)
# cell.add(circle_metal_bottom)
# cell.add(circle_dist)

# circle = gdspy.Round((main_r,0),
#                      4e3/2,
#                      layer = layers['substrate'],
#                      tolerance = 1e1)
# circle_metal_bottom = gdspy.Round((main_r,0),
#                                   4e3/2,
#                                   layer = layers['bottom_metal'],
#                                   tolerance = 1e1)
# circle_metal_top = gdspy.Round((main_r,0),
#                                4e3/2,
#                                layer = layers['top_metal'],
#                                tolerance = 1e1)
# # circle_dist = gdspy.Round((main_r,0),
# #                      4e3/2 - 300 - drill_r,
# #                      layer = layers['dist_layer'],
# #                      tolerance = 1e1)
#
#
# cell.add(circle_metal_top)
# cell.add(circle)
# cell.add(circle_metal_bottom)
# cell.add(circle_dist)
#
# # circle = gdspy.Round((0,-main_r),
#                      4e3/2,
#                      layer = layers['substrate'],
#                      tolerance = 1e1)
# circle_metal_bottom = gdspy.Round((0,-main_r),
#                                   4e3/2,
#                                   layer = layers['bottom_metal'],
#                                   tolerance = 1e1)
# circle_metal_top = gdspy.Round((0,-main_r),
#                                4e3/2,
#                                layer = layers['top_metal'],
#                                tolerance = 1e1)
# # circle_dist = gdspy.Round((0,-main_r),
# #                      4e3/2 - 300 - drill_r,
# #                      layer = layers['dist_layer'],
# #                      tolerance = 1e1)
#
#
# cell.add(circle_metal_top)
# cell.add(circle)
# cell.add(circle_metal_bottom)
# cell.add(circle_dist)
#
# circle = gdspy.Round((0,main_r),
#                      4e3/2,
#                      layer = layers['substrate'],
#                      tolerance = 1e1)
# circle_metal_bottom = gdspy.Round((0,main_r),
#                                   4e3/2,
#                                   layer = layers['bottom_metal'],
#                                   tolerance = 1e1)
# circle_metal_top = gdspy.Round((0,main_r),
#                                4e3/2,
#                                layer = layers['top_metal'],
#                                tolerance = 1e1)
# # circle_dist = gdspy.Round((0,main_r),
# #                      4e3/2 - 300 - drill_r,
# #                      layer = layers['dist_layer'],
# #                      tolerance = 1e1)
#
#
# cell.add(circle_metal_top)
# cell.add(circle)
# cell.add(circle_metal_bottom)
# cell.add(circle_dist)



bind_2 = Hole(shape = 'circle',
                 params = [2.1e3],
                 metallised = True)

bind_2_radius = main_r
bind_2.draw(place = (-bind_2_radius,0),
               angle = 0,
               metal_layers = [layers['top_metal'], layers['bottom_metal']],
               substr_layer = layers['substrate'],
               cell = cell)
bind_2.draw(place = ( bind_2_radius,0),
               angle = 0,
               metal_layers = [layers['top_metal'], layers['bottom_metal']],
               substr_layer = layers['substrate'],
               cell = cell)
bind_2.draw(place = (0,-bind_2_radius),
               angle = 0,
               metal_layers = [layers['top_metal'], layers['bottom_metal']],
               substr_layer = layers['substrate'],
               cell = cell)
bind_2.draw(place = (0,+bind_2_radius),
               angle = 0,
               metal_layers = [layers['top_metal'], layers['bottom_metal']],
               substr_layer = layers['substrate'],
               cell = cell)

#bind holes drawing
bind_hole = Hole(shape = 'circle',
                 params = [binding_holes_r],
                 metallised = True)


bind_hole.draw(place = (bind_w/2,bind_h/2),
               angle = 0,
               metal_layers = [layers['top_metal'], layers['bottom_metal']],
               substr_layer = layers['substrate'],
               cell = cell)
bind_hole.draw(place = (-bind_w/2,bind_h/2),
               angle = 0,
               metal_layers = [layers['top_metal'], layers['bottom_metal']],
               substr_layer = layers['substrate'],
               cell = cell)
bind_hole.draw(place = (bind_w/2,-bind_h/2),
               angle = 0,
               metal_layers = [layers['top_metal'], layers['bottom_metal']],
               substr_layer = layers['substrate'],
               cell = cell)
bind_hole.draw(place = (-bind_w/2,-bind_h/2),
               angle = 0,
               metal_layers = [layers['top_metal'], layers['bottom_metal']],
               substr_layer = layers['substrate'],
               cell = cell)




# central hole drawing
central_hole = Hole(shape='rectangle',
                    params=[hole_w, hole_h],
                    metallised = False)

central_hole.draw(place = (0,0),
                  angle = 0,
                  metal_layers=[layers['top_metal'], layers['bottom_metal']],
                  substr_layer=layers['substrate'], cell=cell)

# mounting holes drawing
smp_place = []
smp_angle = []


cpw_place0 = []
cpw_place1 = []
cpw_place2 = []



for i in [-1,0,1]:
    # smp_place.append((-smp_dist + cpw_offset*(i == 0), inter_smp_dist*i))
    # smp_angle.append(np.pi/2)
    # smp_place.append(( smp_dist - cpw_offset*(i == 0), inter_smp_dist*i))
    # smp_angle.append(np.pi*3/2)
    smp_place.append((inter_smp_dist*i,  smp_dist - cpw_offset*(i == 0)))
    smp_angle.append(0)
    smp_place.append((inter_smp_dist*i, -smp_dist + cpw_offset*(i == 0)))
    smp_angle.append(np.pi)

    # cpw_place0.append((-w/2, inter_cpw_dist*i))
    # cpw_place0.append(( w/2, inter_cpw_dist*i))
    cpw_place0.append((inter_cpw_dist*i,  h/2))
    cpw_place0.append((inter_cpw_dist*i, -h/2))
    #
    # cpw_place1.append((-w1/2, inter_cpw_dist*i))
    # cpw_place1.append(( w1/2, inter_cpw_dist*i))
    cpw_place1.append((inter_cpw_dist*i,  h1/2))
    cpw_place1.append((inter_cpw_dist*i, -h1/2))

    # cpw_place2.append((-w2/2, inter_smp_dist*i))
    # cpw_place2.append(( w2/2, inter_smp_dist*i))
    cpw_place2.append((inter_smp_dist*i,  h2/2))
    cpw_place2.append((inter_smp_dist*i, -h2/2))

i = 0
smp_place.append((-smp_dist, inter_smp_dist*i))
smp_angle.append(np.pi/2)
smp_place.append(( smp_dist, inter_smp_dist*i))
smp_angle.append(np.pi*3/2)

cpw_place0.append((-w/2, inter_cpw_dist*i))
cpw_place0.append(( w/2, inter_cpw_dist*i))
cpw_place1.append((-w1/2, inter_cpw_dist*i))
cpw_place1.append(( w1/2, inter_cpw_dist*i))
cpw_place2.append((-w2/2, inter_smp_dist*i))
cpw_place2.append(( w2/2, inter_smp_dist*i))

cpw_place3 = []
for i in range(0,len(smp_place)):
    mounting = SMP_mounting(place = smp_place[i],
                            angle = smp_angle[i])
    mounting.draw(layer = layers['top_metal'],
                  cell = cell)
    mounting.draw_CPW_connector(S = cpw_s,
                                W = cpw_w,
                                length=connector_l,
                                layer=layers['top_metal'],
                                cell = cell)
    cpw_place3.append(mounting.CPW_place())
    smp_hole = Hole(shape='circle',
                     params = [2e3],
                     metallised = False)

    smp_hole.draw(place=smp_place[i],
                   angle=0,
                   metal_layers=[layers['top_metal'], layers['bottom_metal']],
                   substr_layer=layers['substrate'],
                   cell=cell)

cpw_mask_arr = []
cpw_mask_dist_arr = []
for i in range(0,len(smp_place)):
    print([cpw_place0[i], cpw_place1[i], cpw_place2[i], cpw_place3[i]])
    cpw = CPW(S = cpw_s,
              W = cpw_w,
              path = [cpw_place0[i], cpw_place1[i], cpw_place2[i], cpw_place3[i]],
              curve_r = cpw_curvature)
    res = cpw.draw_mask(layer = layers['top_metal'],
             cell = cell)
    cpw_mask_arr.append(res[0])
    cpw_mask_dist_arr.append(res[1])


cpw_mask = cpw_mask_arr[0]
cpw_mask_dist = cpw_mask_dist_arr[0]
for i in cpw_mask_arr:
    cpw_mask = gdspy.boolean(cpw_mask,i,'or')
for i in cpw_mask_dist_arr:
    cpw_mask_dist = gdspy.boolean(cpw_mask_dist,i,'or')

layer_boolean(cell, layers['top_metal'], cpw_mask, 'not')
layer_boolean(cell, layers['dist_layer'], cpw_mask_dist, 'not')

# merging(cell,2)
# draw_vias(d = drill_r*2,
#           a_vec=(0,via_hole_r*2+tech_dist),
#           b_vec=(via_hole_r*2+tech_dist,0),
#           layer=layers['via_holes'],
#           cell=cell)
#
# layer_clear(cell = cell,
#             layer_N=24)
# saving section
lib.write_gds(filename + '.gds')
cell.write_svg(filename + '.svg')
gdspy.LayoutViewer() # Display all cells using the internal viewer.




