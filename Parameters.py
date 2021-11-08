from CPW_calculator import *
unit = 1e-6
precision = 1e-9

main_r = 50*1e3/2
central_hole_w = 10.2e3
central_hole_h = 10.2e3

substrate_thickness = 503
metal_thickness = 18

via_hole_r = 300
binding_holes_r = 3.5e3



cpw_b = 1.2e3

# cpw_s, cpw_w = find_50(cpw_b)

# cpw_s = 130
# cpw_w = 650

cpw_s = 222
cpw_w = 604

connector_l = 4e3
smp_number = 12
cpw_curvature = 3.5e3
hole_w = 10.2*1e3
hole_h = 5.2*1e3
drill_radius = 0.5*1e3  # radius of cutting inrtument


# drill_r = 300/2 # Zero for nonvias one
drill_r = 300/2
tech_dist = 125

inter_cpw_dist = 2.5e3
cpw_offset = 0e3

inter_smp_dist = 6.5e3
smp_dist = 19.5e3

w = 10.2e3/2+tech_dist
h = 5.2e3/2+tech_dist


w1 = 16e3
h1 = 16e3

w2 = 22e3
h2 = 22e3

bind_h = 20e3
bind_w = 20e3