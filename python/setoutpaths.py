import os

def filefind(experiment, rootdir='default_output'):
    # Example:
    # rootdir = 'resultsbaseline/' + Interpolant + '/'
    figure_directory = rootdir + 'figures/' +'experiment'+str(experiment)+'/'
    if not os.path.exists(figure_directory) or not os.path.exists(rootdir + 'figures/'):
        os.makedirs(figure_directory)
    # Output filename stored as outfilename.NPZ in OUT/ folder
    out_directory = rootdir + 'out/'+'experiment'+str(experiment)+'/'
    previous_out_directory = rootdir + 'out/'+'experiment'+str(experiment-1)+'/'
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    outfilename = out_directory+'results'
    previous_outfilename = previous_out_directory+'results'
    return figure_directory, outfilename, previous_outfilename, rootdir
