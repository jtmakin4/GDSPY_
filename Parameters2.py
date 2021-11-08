# from CPW_calculator import *

# служебные переменные
unit = 1e-6
precision = 1e-9
tech_dist = 125
tech_layer = 24

layers = {
    'bottom_metal': 0,
    'substrate': 1,
    'top_metal': 2,
    'via_holes': 16
}

# dimensions of pcb
main_d = 50e3
main_w = 50e3
main_h = 50e3

# parameters of central hole
central_hole_w = 10.2e3
central_hole_h = 10.2e3
instrument_d = 1*1e3  # radius of cutting inrtument
line_number = 12
inter_cpw_dist = 2.5e3

# materials parameters
substrate_thick = 503
metal_thick = 18*2
substrate_eps = 10.2

via_d = 50

# assembly holes' parameters
assembly_hole_d = 3.5e3
assembly_hole_h = 20e3
assembly_hole_w = 20e3

# cpws' parameters
cpw_b = 1.2e3
cpw_s = 130
cpw_w = 650
cpw_g = 300 # расстояние до виасов
cpw_r = 3.5e3

# smp mounting places' parameters
smp_connect_l = 3e3
inter_smp_dist = 6.3e3
smp_dist = 19.5e3
cpw_offset = 0e3

# drill_r = 300/2 # Zero for nonvias one
# drill_r = 300/2

# rectangles of cpw drawing
draw_tier_w = [central_hole_w+tech_dist, 16e3, 22e3]
draw_tier_h = [central_hole_h+tech_dist, 16e3, 22e3]

if __name__ == '__main__':
    print("data input")
    # записать функцию для перезаписи данных