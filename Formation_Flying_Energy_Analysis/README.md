Ideas for the implementation:
* Add functionality to queue multiple simulaiton runs with different configurations. [WILL THIS ACTUALLY BE USEFUL THOUGH?]
    - the easiest way would probably be to have multiple config files, one for each run. The configs don't need to be split up into Basilisk and Skyfield here, so only 1 file per run

* Separate simulation and plotting runs. 
    - I have learned that it can be cumbersome to constantly change 'default.yaml' when changing between running simulations, and skipping simulation to plot. 
    - At least make it possible to run the plotting script isolated from simulation 'python3 plotting.py' 
    - Since there is going to be multiple simulation config files, I think it is best to move the plotting settings to a separate config file. This is to avvoid a lot of duplicate settings between queued simuilation configurations, where plotting isn't relevant. 

* Implement measures to reduce simulation data output size and to avvoid multiple processing 
    - I imagine the runs are going to produce huge amounts of data because of the long run durations. Having the ability to only store data every X seconds can therefore be very practical to reduce filesizes 
        - Alternatively, if I don't want to store less data in fear of missing quick responses, I can also save the full datafile, and export downsampled versions that will be used for plotting/post-processing. This will increase the amount of data stored, but will allow me to experiment with different sample step sizes without having to run multiple simulations.
    - Implement functionality to avvoid processing the data the same way multiple times. 
        - For ex: In the comparison simulator, the ECI -> RTN transformations are performed every time the plot is generated. This is extremely inefficient
        - Therefor, whenever it is needed, perform the processing step(s) ONCE, the store the processed data. Checks can be poerformed on later runs to check if processing has been done before. 




NB! The below flow assumes that it will be usefull to queue simulations. It might not be actually...




* High-level program flow should be something like:
    1. User defines XX config files, one for each simulation configuration that shall be run (effectively defining a simulation queue)
        - The config files should then ONLY define relevant simulation parameters, NOT plotting parameters
    2. The user can run <python3 run_formation_energy_consumption_queue> (or something) which will run all the different configurations and outputs data
        - IDEA: For each set of simulations/queue, create an output data folder that contains oneh5 datafile for each simulation output. The folder can just be named <timestamp>, or have a descriptive name like <control_law_pointing_acc>. The path to this folder can then be used as a standard input parameter to all plotting functions
            - If there is only one simulation configuration queued, skip creating a folder, and simply add the file to the output directory.
        - IDEA: In the config for each run, there should be (unique) field called something like <short_desc>/<legend>/<desc_id> that contains a str that very briefly describes the configuration (similar to 'Bsk: MSIS atm, SH4, Drag, leader A_D=0.12m2' in plot.py). This will be the legend for the corresponding data of that run. 
            - I imagine that datafiles in the datafolder for one entire queue will just be named run1, run2, etc (or similar). It will then be simplest if data is plotted in that order. HOWEVER, if unique, <desc_id> can be used to define a custom plotting order by having an optional plotting order input to plotting functions, like this: [<desc_id_1>, <desc_id_3>, <desc_id_2>]
                - NOTE: If I want to add this feature, I must implement a validation step that checks for unique <desc_id>s
                - From experience, I know that the legends can change later to create the best possible plots. Therefore, I must also implement some way to change the <desc_id> value belonging to some data after the simulation is complete.
    3. The user defines a single config file that specifies:
        - Output plot size XXX x YYY (apply to all files)
        - data directory path (or simply assume that the directory is in a constant folder, in which case only define the data directory name)
        - Optional: custom plotting order (using <desc_id>)
        - etc.
    4. The user can run <python3 plot_formation_energy_consumption_data> (or something) to plot all the output data