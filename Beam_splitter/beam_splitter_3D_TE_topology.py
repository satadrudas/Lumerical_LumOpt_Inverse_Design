######## IMPORTS ########
# General purpose imports
import numpy as np
import os
import sys
import scipy as sp

# Optimization specific imports
from lumopt import CONFIG
from lumopt.geometries.topology import TopologyOptimization2D, TopologyOptimization3DLayered
from lumopt.utilities.load_lumerical_scripts import load_from_lsf
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimization import Optimization
from lumopt.optimizers.generic_optimizers import ScipyOptimizers
from lumopt.utilities.wavelengths import Wavelengths

###
from lumopt.optimizers.adaptive_gradient_descent import AdaptiveGradientDescent 
from lumopt.optimizers.fixed_step_gradient_descent import FixedStepGradientDescent 
from lumopt.optimizers.scipy_basin_hopping import ScipyBasinHopping 

cur_path = os.path.dirname(os.path.abspath(__file__))

######## RUNS TOPOLOGY OPTIMIZATION OF A 3D STRUCTURE ########
def runSim(params, eps_bg, eps_wg, x_pos, y_pos, z_pos, filter_R, min_feature_size, working_dir, beta = 1):

    ######## DEFINE A 3D TOPOLOGY OPTIMIZATION REGION ########
    geometry = TopologyOptimization3DLayered(params=params, eps_min=eps_bg, eps_max=eps_wg, x=x_pos, y=y_pos, z=z_pos, filter_R=filter_R, min_feature_size=min_feature_size, beta=beta)

    ######## DEFINE FIGURE OF MERIT ########
    # The base simulation script defines a field monitor named 'fom' at the point where we want to modematch to the fundamental TE mode
    fom1 = ModeMatch(monitor_name = 'fom_top', mode_number = 'Fundamental TE mode', direction = 'Forward', norm_p = 2, target_fom=0.5)
    fom2 = ModeMatch(monitor_name = 'fom_bottom', mode_number = 'Fundamental TE mode', direction = 'Forward', norm_p = 2, target_fom=0.5)
    
   
    ######## DEFINE OPTIMIZATION ALGORITHM ########
    optimizer = FixedStepGradientDescent(max_dx=1,max_iter=1000,all_params_equal=False,noise_magnitude=0.1,scaling_factor=1)#ScipyOptimizers(max_iter=1000, method='L-BFGS-B', scaling_factor=1, pgtol=1e-6, ftol=1e-4, scale_initial_gradient_to=0.25)


    ######## LOAD TEMPLATE SCRIPT AND SUBSTITUTE PARAMETERS ########
    script = load_from_lsf(os.path.join(cur_path, 'beam_splitter_3D.lsf'))
    
    ## Here, we substitute the size of the optimization region to properly scale the simulation domain 
    size_x = max(x_pos) - min(x_pos)
    script = script.replace('opt_size_x=4e-6','opt_size_x={:1.6g}'.format(size_x))
    
    size_y = max(y_pos) - min(y_pos)
    script = script.replace('opt_size_y=4e-6','opt_size_y={:1.6g}'.format(2*size_y))

    wavelengths1 = Wavelengths(start = 1500e-9, stop = 1600e-9, points = 2)
    opt1 = Optimization(base_script=script, wavelengths = wavelengths1, fom=fom1, geometry=geometry, optimizer=optimizer, use_deps=False, hide_fdtd_cad=False, plot_history=True, store_all_simulations=False, save_global_index=True)
    opt2 = Optimization(base_script=script, wavelengths = wavelengths1, fom=fom2, geometry=geometry, optimizer=optimizer, use_deps=False, hide_fdtd_cad=False, plot_history=True, store_all_simulations=False, save_global_index=False)

    ######## PUT EVERYTHING TOGETHER AND RUN ########
    opt = opt1+opt2
    #opt.continuation_max_iter = 40 #< How many iterations per binarization step (default is 20)
    opt.run(working_dir = working_dir)
    
    
    
    
    
    
if __name__ == '__main__':
    size_x = 4000
    size_y = 4000
    size_z = 220
    resolution=20
    
    filter_R = 150e-9
    min_feature_size = filter_R
    
    eps_wg = 3.48**2#2.8**2
    eps_bg = 1#1.44**2
    
    if len(sys.argv) > 2 :
        size_x = int(sys.argv[1])
        size_y = int(sys.argv[2])
        filter_R = int(sys.argv[3])*1e-9
        print(size_x,size_y,filter_R)

    x_points=int(size_x/resolution)+1
    y_points=int(size_y/resolution)+1
    z_points=int(size_z/resolution)+1    

    x_pos = np.linspace(-size_x/2*1e-9,size_x/2*1e-9,x_points)
    y_pos = np.linspace(-size_y/2*1e-9,size_y/2*1e-9,y_points)
    z_pos = np.linspace(-size_z/2*1e-9,size_z/2*1e-9,z_points)
    

    start_from_2d_result = False

    if start_from_2d_result:
    
        ## First, we specify in which path to find the 2d optimization
        path2d = os.path.join(cur_path,  'splitter_2D_TE_topo_x{:04d}_y{:04d}_f{:04d}_0'.format(size_x,size_y,int(filter_R*1e9)))
        
        ## Next, we load the log-file to figure out which iteration we should load from. It is usually best to load something mid-way through the binarization
        convergence_data = np.genfromtxt(os.path.join(path2d, 'convergence_report.txt'), delimiter=',')
        
        ## Find the first row where the beta value (3rd column) is larger than beta_threshold (here 10)
        beta_threshold = 10
        beta_filter = convergence_data[:,2]>beta_threshold
        convergence_data = convergence_data[beta_filter,:]  #< Trim data down
        iteration_number = int(convergence_data[0,0]-2)
        
        ## Load the 2d parameter file for the specified iteration
        geom2d = TopologyOptimization2D.from_file(os.path.join(path2d, 'parameters_{}.npz'.format(iteration_number) ), filter_R=filter_R, eta=0.5)
              
        startingParams = geom2d.last_params     #< Use the loaded paramters as starting parameters for 3d
        startingBeta   = geom2d.beta            #< Also start with the same beta value. One could start with 1 as well.
    
    else:
        startingBeta   = 1

        startingParams = None
#        startingParams = 0.5*np.ones((x_points,y_points))   #< Start with the domain filled with (eps_max+eps_min)/2
#        startingParams = np.ones((x_points,y_points))       #< Start with the domain filled with eps_max
#        startingParams = np.zeros((x_points,y_points))      #< Start with the domain filled with eps_min  
#        startingParams = np.random.randint(low=0, high=2, size=(x_points,y_points), dtype=np.int64)         
        
    working_dir = os.path.join(cur_path, 'beam_splitter_3D_TE_topo_x{:04d}_y{:04d}_f{:04d}'.format(size_x,size_y,int(filter_R*1e9)))
    runSim(startingParams, eps_bg, eps_wg, x_pos, y_pos, z_pos, filter_R=filter_R, min_feature_size=min_feature_size, working_dir=working_dir, beta = startingBeta)

    
    
