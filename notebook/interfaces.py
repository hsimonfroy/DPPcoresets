import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from ipywidgets import interactive
from pathlib import Path
from IPython.display import display




#########################
# plot 3D from function #
#########################
def plot3d_func(ax, func, nb_discr=100, cube_center=0, cube_halfsize=1.5):
    """
    e.g.:
        plt.figure(figsize=(8,4))
        ax = plt.subplot(121, projection="3d")
        plot3d_func(ax, my_pdf, 100)
    """
    x, y = np.linspace(cube_center-cube_halfsize, cube_center+cube_halfsize, nb_discr), np.linspace(cube_center-cube_halfsize, cube_center+cube_halfsize, nb_discr)
    xx, yy = np.meshgrid(x, y)
    xy = np.array([xx, yy]).transpose(1,2,0)
    zz =  func(xy.reshape(-1,2)).reshape(nb_discr,nb_discr)
    surf = ax.plot_surface(xx, yy, zz, cmap="viridis")



########################
# save plots interface #
########################
def save_current_subplots(fig_directory, fig_name):
    # set figure directory 
    Path(fig_directory).mkdir(parents=True, exist_ok=True)
    # save figure and its subplots
    plt.savefig(fig_directory + fig_name)
    for i_ax, ax in enumerate(plt.gcf().axes):
        extent = ax.get_tightbbox().transformed(plt.gcf().dpi_scale_trans.inverted()).padded(2/72) # add 2 pts of padding
        plt.savefig(fig_directory + fig_name + "_ax" + str(i_ax), bbox_inches=extent)
    print("figure saved")


def set_plotting_options(use_TeX, font_size):
    # reset plotting options, in case they has been coincidentaly altered
    mpl.rcParams.update(mpl.rcParamsDefault)
    # set plotting options
    params = {'axes.titlesize': font_size,
              'axes.labelsize': font_size,
              'xtick.labelsize': font_size,
              'ytick.labelsize': font_size,
              'legend.fontsize': font_size,
              'text.usetex': use_TeX} # 'ps.useafm', 'pdf.use14corefonts'
    plt.rcParams.update(params)

# TODO: plot_figure in independant function

def get_save_plots_interface(my_plot, 
                    fig_width=6.40, fig_height=4.80, use_TeX=False, font_size=10,
                    fig_directory="./figs/", fig_name="FIG"):
    # these default parameters are matplotlib default options

    # construct plot button
    def plot_figure(fig_width=fig_width, fig_height=fig_height, use_TeX=use_TeX, font_size=font_size):
        set_plotting_options(use_TeX, font_size)
        plt.figure(figsize=(fig_width, fig_height))
        my_plot()
        plt.tight_layout()
        display(plt.gcf())

    plot_button = interactive(plot_figure, {"manual":True, "manual_name":"plot figure"},
                              fig_width=(4.0, 20.0),
                              fig_height=(3.0, 15.0),
                              use_TeX=False, 
                              font_size=(6,16))
    
    # construct save button
    save_button = interactive(save_current_subplots, {"manual":True, "manual_name":"save figure"},
                          fig_directory=fig_directory, 
                          fig_name=fig_name)
    display(plt.gcf())
    return plot_button, save_button



############################
# save variables interface #
############################
def save_variables_as_dict(variable_names="variable1, variable2", output_directory="./npys/", output_name="OUTPUT", save_separately=False):
    variable_names_list = [unstriped.strip() for unstriped in variable_names.split(",")]

    if save_separately:
        # save each variable separately in a dict 
        # e.g. {name1:value1} in file OUTPUT_name1, {name2:value2} in file OUTPUT_name2
        for variable_name in variable_names_list:
            dict_to_save = dict()
            variable_value = globals()[variable_name] # access global variables
            dict_to_save[variable_name] = variable_value
            output_fullname = output_directory + output_name + "_" + variable_name
            np.save(output_fullname, dict_to_save)
            print(f"Saved {output_fullname}.npy")
            print(f"""Type `np.load("{output_fullname}.npy", allow_pickle=True).item()` to load""")

    else:
        # save all variables in a dict
        # e.g. {name1:value1, name2_value2} in file OUTPUT
        dict_to_save = dict()
        for variable_name in variable_names_list:
            variable_value = globals()[variable_name] # access global variables
            dict_to_save[variable_name] = variable_value
        output_fullname = output_directory + output_name
        np.save(output_fullname, dict_to_save)
        print(f"Saved {output_fullname}.npy")
        print(f"""Type `np.load("{output_fullname}.npy", allow_pickle=True).item()` to load""")

def get_save_variables_interface(variable_names="variable1, variable2", output_directory="./npys/", output_name="OUTPUT", save_separately=False):
    # construct save variables button
    save_variables_button = interactive(save_variables_as_dict, {"manual":True, "manual_name":"save variables"},
                                        variable_names=variable_names,
                                        output_directory=output_directory, 
                                        output_name=output_name,
                                        save_separately=save_separately)
    return save_variables_button