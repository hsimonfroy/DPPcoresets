import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from ipywidgets import interactive, fixed, IntSlider, FloatSlider
from pathlib import Path
from IPython.display import display
# accesses global variables dict of __main__, see also vars(__main__). 
# globals() call would provide global variables of this module scope only.
from __main__ import __dict__




#########################
# plot 3D from function #
#########################
def plot3d_func(ax, func, nb_discr=100, cube_center=0, cube_halfsize=1.5):
    """
    e.g.:
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
    # assert fig_name does not include further directories path or other improper chars
    forbidden_chars = ["/", "\\", ":", "\"", "?"] # backslash is "\\" or chr(92)
    assert all([forbidden_char not in fig_name for forbidden_char in forbidden_chars]), f"output_name should not contain these chars: {forbidden_chars}"
    # set output directory 
    Path(fig_directory).mkdir(parents=True, exist_ok=True)

    # if an extension is provided in the name, it is selected and appended to subplots names
    # e.g. "FIG.jpg" would save "FIG.jpg" figure and "FIG_ax1.jpg", "FIG_ax2.jpg" subplots
    fig_name_stem_suffix = Path(fig_name)
    fig_name_stem, fig_name_suffix = fig_name_stem_suffix.stem, fig_name_stem_suffix.suffix

    # save figure and its subplots
    plt.savefig(fig_directory + fig_name)
    axes_list = plt.gcf().axes
    if axes_list:
        for i_ax, ax in enumerate(axes_list):
            extent = ax.get_tightbbox().transformed(plt.gcf().dpi_scale_trans.inverted()).padded(0.1/2.54) # add 0.1 cm of padding (matplotlib unit is inches)
            plt.savefig(fig_directory + fig_name_stem + "_ax" + str(i_ax+1) + fig_name_suffix, bbox_inches=extent)
        print(f"Saved figure and its {i_ax+1} subplots in {fig_directory} folder.")
    else:
        print(f"Saved figure in {fig_directory} folder.")    

def set_plotting_options(use_TeX, font_size):
    # reset plotting options, in case they has been coincidentaly altered
    mpl.rcParams.update(mpl.rcParamsDefault)
    # set plotting options
    params = {'axes.titlesize': font_size,
              'axes.labelsize': font_size,
              'xtick.labelsize': font_size,
              'ytick.labelsize': font_size,
              'legend.fontsize': font_size,
              'text.usetex': use_TeX} # see also 'ps.useafm', 'pdf.use14corefonts'
    plt.rcParams.update(params)

def plot_figure(my_plot, fig_width=6.4, fig_height=4.8, use_TeX=False, font_size=10):
    # default plotting parameters are matplotlib default options
    set_plotting_options(use_TeX, font_size)
    plt.close() # close potential previous figures to not saturate cache
    plt.figure(figsize=(fig_width, fig_height))
    my_plot()
    plt.tight_layout()
    display(plt.gcf()) # use display() instead of plt.show(), because the latter close figure automatically which doesn't allow to save it later. However, it is better to not forget to close figure later.

def get_save_plots_interface(my_plot, 
                    fig_size=(6.4, 4.8), use_TeX=False, font_size=10,
                    fig_directory="./figs/", fig_name="FIG"):
    # default plotting parameters are matplotlib default options

    # construct plot figure button
    fig_width, fig_height = fig_size
    plot_fig_button = interactive(plot_figure, {"manual":True, "manual_name":"plot figure"},
                              my_plot=fixed(my_plot),
                              fig_width=FloatSlider(min=4., max=20., step=.2, value=fig_width, continuous_update=False),
                              fig_height=FloatSlider(min=3., max=15., step=.2, value=fig_height, continuous_update=False),
                              use_TeX=use_TeX, 
                              font_size=IntSlider(min=6, max=16, step=1, value=font_size, continuous_update=False))
    
    # construct save figure button
    save_fig_button = interactive(save_current_subplots, {"manual":True, "manual_name":"save figure"},
                          fig_directory=fig_directory, 
                          fig_name=fig_name)
    
    return plot_fig_button, save_fig_button

def save_plots_interface(my_plot, 
                    fig_size=(6.4, 4.8), use_TeX=False, font_size=10,
                    fig_directory="./figs/", fig_name="FIG", plot_on_call=False):
    # default plotting parameters are matplotlib default options

    # plot figure on the call of this function with passed parameters
    if plot_on_call:
        fig_width, fig_height = fig_size
        plot_figure(my_plot, fig_width, fig_height, use_TeX, font_size)

    # get plot figure and save figure button and display them
    plot_fig_button, save_fig_button = get_save_plots_interface(my_plot, 
                                                                fig_size, use_TeX, font_size, 
                                                                fig_directory, fig_name)
    display(plot_fig_button, save_fig_button)




############################
# save variables interface #
############################
def save_variables_as_dict(variable_names="var1, var2", output_directory="./npys/", output_name="OUTPUT", save_separately=False):
    # assert output_name does not include further directories path or other improper chars
    forbidden_chars = ["/", "\\", ":", "\"", "?"] # backslash is "\\" or chr(92)
    assert all([forbidden_char not in output_name for forbidden_char in forbidden_chars]), f"output_name should not contain these chars: {forbidden_chars}"
    # set output directory 
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # split along "," and strip potential extra-spaces 
    variable_names_list = [unstriped.strip() for unstriped in variable_names.split(",")]

    if save_separately:
        # save each variable separately in a dict 
        # e.g. {name1:value1} in file OUTPUT_name1, {name2:value2} in file OUTPUT_name2
        for variable_name in variable_names_list:
            dict_to_save = dict()
            variable_value = __dict__[variable_name] # access global variable
            dict_to_save[variable_name] = variable_value
            output_fullname = output_directory + output_name + "_" + variable_name
            np.save(output_fullname, dict_to_save)
            print(f"Saved {output_fullname}.npy")
        print(f"Saved variables in {output_directory} folder. To load last variable for instance, type:")
        loading_hint_str = f"""```\n{output_name} = np.load("{output_fullname}.npy", allow_pickle=True).item()\n"""
        loading_hint_str += variable_name + f", = {output_name}.values()\n```" 
        # In `variable, = dict.values()` the comma is crucial to handle single entry dict
        print(loading_hint_str)

    else:
        # save all variables in a dict
        # e.g. {name1:value1, name2_value2} in file OUTPUT
        dict_to_save = dict()
        for variable_name in variable_names_list:
            variable_value = __dict__[variable_name] # access global variable
            dict_to_save[variable_name] = variable_value
        output_fullname = output_directory + output_name
        np.save(output_fullname, dict_to_save)
        print(f"Saved variables in {output_directory} folder. To load, type:")
        loading_hint_str = f"""```\n{output_name} = np.load("{output_fullname}.npy", allow_pickle=True).item()\n"""
        for variable_name in variable_names_list:
            loading_hint_str += variable_name + ", "
        loading_hint_str = loading_hint_str + f"= {output_name}.values()\n```"
        # In `variable, = dict.values()` the comma is crucial to handle single entry dict
        print(loading_hint_str)

def get_save_variables_interface(variable_names="var1, var2", output_directory="./npys/", output_name="OUTPUT", save_separately=False):
    # construct save variables button
    save_variables_button = interactive(save_variables_as_dict, {"manual":True, "manual_name":"save variables"},
                                        variable_names=variable_names,
                                        output_directory=output_directory, 
                                        output_name=output_name,
                                        save_separately=save_separately)
    return save_variables_button

def save_variables_interface(variable_names="var1, var2", output_directory="./npys/", output_name="OUTPUT", save_separately=False, save_on_call=False):
    # save variables on the call of this function with passed parameters
    if save_on_call:
        save_variables_as_dict(variable_names, output_directory, output_name, save_separately)
    # get save variables button and display it
    save_variables_button = get_save_variables_interface(variable_names, output_directory, output_name, save_separately)
    display(save_variables_button)













