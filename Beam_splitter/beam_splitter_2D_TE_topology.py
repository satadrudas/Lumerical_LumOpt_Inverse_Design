######## IMPORTS ########
# General purpose imports
import numpy as np
import os
import sys
import scipy as sp

# Optimization specific imports
from lumopt import CONFIG
from lumopt.geometries.topology import TopologyOptimization2D
from lumopt.utilities.load_lumerical_scripts import load_from_lsf
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimization import Optimization,SuperOptimization
from lumopt.optimizers.generic_optimizers import ScipyOptimizers
from lumopt.utilities.wavelengths import Wavelengths

###
from lumopt.optimizers.adaptive_gradient_descent import AdaptiveGradientDescent 
from lumopt.optimizers.fixed_step_gradient_descent import FixedStepGradientDescent 
from lumopt.optimizers.scipy_basin_hopping import ScipyBasinHopping 

cur_path = os.path.dirname(os.path.abspath(__file__))

######## RUNS TOPOLOGY OPTIMIZATION OF A 2D STRUCTURE ########
def runSim(params, eps_min, eps_max, x_pos, y_pos, size_x, size_y,filter_R, working_dir,beta):

    ######## DEFINE A 2D TOPOLOGY OPTIMIZATION REGION ########
    geometry = TopologyOptimization2D(params=params, eps_min=eps_min, eps_max=eps_max, x=x_pos, y=y_pos, z=0, filter_R=filter_R, min_feature_size=filter_R,beta=beta)

    ######## DEFINE FIGURE OF MERIT ########
    # The base simulation script defines a field monitor named 'fom' at the point where we want to modematch to the fundamental TE mode
    fom1 = ModeMatch(monitor_name = 'fom_top', mode_number = 'Fundamental TE mode', direction = 'Forward', norm_p = 2, target_fom=0.50)
    fom2 = ModeMatch(monitor_name = 'fom_bottom', mode_number = 'Fundamental TE mode', direction = 'Forward', norm_p = 2, target_fom=0.50)

    ######## DEFINE OPTIMIZATION ALGORITHM ########
    optimizer=ScipyOptimizers(max_iter=1000, method='L-BFGS-B', scaling_factor=1, pgtol=1e-6, ftol=1e-5, scale_initial_gradient_to=1.0)

    ######## LOAD TEMPLATE SCRIPT AND SUBSTITUTE PARAMETERS ########
    script = load_from_lsf(os.path.join(cur_path, 'beam_splitter_2D.lsf'))
    script = script.replace('opt_size_x=6e-6','opt_size_x={:1.6g}'.format(size_x))
    script = script.replace('opt_size_y=5e-6','opt_size_y={:1.6g}'.format(size_y))  #< Factor 2 is because of symmetry in y-direction

    ######## SETTING UP THE OPTIMIZER ########
    wavelengths = Wavelengths(start = 1530e-9, stop = 1570e-9, points = 10)
        
    opt1 = Optimization(base_script=script, wavelengths = wavelengths, fom=fom1, geometry=geometry, optimizer=optimizer, use_deps=False, hide_fdtd_cad=False, plot_history=False, store_all_simulations=False, save_global_index=True)
    opt2 = Optimization(base_script=script, wavelengths = wavelengths, fom=fom2, geometry=geometry, optimizer=optimizer, use_deps=False, hide_fdtd_cad=False, plot_history=False, store_all_simulations=False, save_global_index=False)

    
    
    opt_total =[opt1, opt2]
    weights=[0.5,0.5]
    opt=SuperOptimization(optimizations=opt_total, weights=weights)
    #opt.continuation_max_iter = 20 #< How many iterations per binarization step (default is 20)

    ######## RUN THE OPTIMIZER ########
    #opt=opt1+opt2
    #opt.continuation_max_iter = 20 #< How many iterations per binarization step (default is 20)
    opt.run(working_dir = working_dir)
   
    
    
if __name__ == '__main__':
    size_x = 6000
    size_y = 5000
    filter_R = 150e-9

    x_points=int(size_x/20)+1
    y_points=int(size_y/20)+1
    eps_wg = 3.48**2
    eps_bg = 1.0**2
    x_pos = np.linspace(-size_x/2*1e-9,size_x/2*1e-9,x_points)
    y_pos = np.linspace(-size_y/2*1e-9,size_y/2*1e-9,y_points)

    ## We need to pick an initial condition. Many different options:
    params = 0.5*np.ones((x_points,y_points))     #< Start with the domain filled with (eps_wg+eps_bg)/2
    #params = np.ones((x_points,y_points))        #< Start with the domain filled with eps_wg
    #params = np.zeros((x_points,y_points))       #< Start with the domain filled with eps_bg    
    #params = None                                #< Use the structure defined in the project file as initial condition
    #params=np.random.randint(low=0, high=2, size=(x_points,y_points), dtype=np.int64)

    working_dir = 'beam_splitter_2D_TE_topo_x{:04d}_y{:04d}_f{:04d}'.format(size_x,size_y,int(filter_R*1e9))
    runSim(params, eps_bg, eps_wg, x_pos, y_pos, size_x*1e-9, size_y*1e-9, filter_R, working_dir=working_dir, beta=1)
