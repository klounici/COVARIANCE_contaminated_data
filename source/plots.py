#!/usr/bin/env python
"""
Mains function for plots

Authors: Karim Lounici and Grégoire Pacreau
"""

from math import exp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import pandas
import numpy
import json
from source.auxiliary_functions import compute_error
import os

# width of a latex document
WIDTH = 470

def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width
        
    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27

    golden_ratio = (5**.5 - 1) / 2

    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

def show_available(save_folder="./results/"):
    filelist = os.listdir(save_folder)
    available = {}
    for i, filename in enumerate(filelist):
        if filename.endswith(".pkl") and not filename.startswith("exectime"):
            diconame = filename.replace('.pkl', '"}')
            diconame = diconame.replace('_', '","')
            diconame = diconame.replace('=', '":"')
            diconame = '{"' + diconame
            available[i] = eval(diconame)
    return pandas.DataFrame(available).transpose()

def get_errors_from_cov_dict(dico, index=None):
    truth = dico["truth"]
    methods = list(dico.keys())
    methods.remove("truth")
    errors = {}
    for method in methods:
        errors[method] = {}
        for epsilon in dico[method].keys():
            errors[method][epsilon] = []
            for i, true_cov in enumerate(truth[epsilon]):
                try:
                    error = compute_error(true_cov, dico[method][epsilon][i], index=index)
                    errors[method][epsilon].append(error)
                except:
                    pass
    return errors

def filename_to_dico(f):
    if f.endswith(".pkl"):
        f = f.replace(".pkl", "")
    f = '{"'+ f + '"}'
    f = f.replace('_', '","')
    f = f.replace('=', '":"')
    return json.loads(f)

def test_all_available(f1, f2):
    caracs_f1 = list(f1.keys())
    caracs_f2 = list(f2.keys())

    caracs_f1.remove("meth")
    caracs_f2.remove("meth")
    for c1 in caracs_f1:
        if c1 not in caracs_f2:
            return False
        elif f1[c1] != f2[c1]:
            return False
    return True

def test_list(f1, f2):
    caracs_f1 = list(f1.keys())
    caracs_f2 = list(f2.keys())

    meths = f1["meth"]

    caracs_f1.remove("meth")
    caracs_f2.remove("meth")
    if f2["meth"] in meths:
        for c1 in caracs_f1:
            if c1 not in caracs_f2:
                return False
            elif f2[c1] == "None":
                pass
            elif f1[c1] != f2[c1]:
                return False
    else:
        return False
    return True

def plot_dico(nsample, dim, erank, nexp, cont, intensity=1, cont_type="gauss", meth="all-available", replace=True, index=None, folder=None, save_folder="./results/"):
    filename = "nsample={}_dim={}_erank={}_nexp={}_cont={}_meth={}_intensity={}_cont-type={}".format(nsample, dim, erank,nexp,cont,meth, intensity, cont_type)
    file_carac = filename_to_dico(filename)

    if cont == "adversarial":
        # deleting irrelevant options for adversarial contamination 
        del file_carac["intensity"]
        del file_carac["cont-type"]

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8
    }

    plt.rcParams.update(tex_fonts)  

    figurelist = os.listdir("./figures/")
    if filename+'_index='+str(index)+'.pdf' in figurelist and not replace:
        # If the image already exists, just plot it
        plt.imshow(mpimg.imread('./figures/'+filename+'_index='+str(index)+'.pdf'))
    else:
        if meth == 'all-available':
            # fetches all available dicts with these parameters and blends methods
            dicts = []
            for f in os.listdir(save_folder):
                if f == ".DS_Store" or f == '.ipynb_checkpoints':
                    pass
                elif f.startswith("exectime") or f.endswith('.ipynb'):
                    pass
                else:
                    new_dico = filename_to_dico(f)
                    if test_all_available(file_carac, new_dico):
                        with open(save_folder + f, 'rb') as file:
                            expe_res = pickle.load(file)
                            error_dico = get_errors_from_cov_dict(expe_res, index=index)
                            dicts.append(error_dico)
            res_dico = {}
            for d in dicts:
                res_dico.update(d)

        elif isinstance(meth, list):
            dicts = []
            for f in os.listdir(save_folder):
                if f == '.DS_Store' or f == '.ipynb_checkpoints':
                    pass
                elif f.startswith("exectime") or f.endswith('.ipynb'):
                    pass
                else:
                    new_dico = filename_to_dico(f)
                    if test_list(file_carac, new_dico):
                        with open(save_folder + f, 'rb') as file:
                            expe_res = pickle.load(file)
                            error_dico = get_errors_from_cov_dict(expe_res, index=index)
                            dicts.append(error_dico)
            res_dico = {}
            for d in dicts:
                res_dico.update(d)
        else:
            # Otherwise, select the specific method looked after
            assert filename+'.pkl' in os.listdir(save_folder), "figure must correspond to preexisting experiment"
            with open(save_folder+filename+'.pkl', 'rb') as file:
                expe_res = pickle.load(file)
                res_dico = get_errors_from_cov_dict(expe_res, index=index)
        
        # computing means and stds
        acquired = []
        
        if isinstance(meth, list):
            methods = []
            for m in meth:
                if m in res_dico.keys():
                    methods.append(m)
        else:
            methods = list(res_dico.keys())
        epsilons = res_dico[methods[0]].keys()
        means = []
        stds = []
        for m in methods:
            temp_m = []
            temp_s = []
            for e in epsilons:
                temp_m.append(numpy.mean(res_dico[m][e]))
                temp_s.append(numpy.std(res_dico[m][e]))
            means.append(temp_m)
            stds.append(temp_s)
        means = numpy.array(means)
        stds = numpy.array(stds)

        known_methods = ['classical', 'DDCMV95', 'DDCMV90', 'tailMV', 'TSGS', 'DI', 'oracleMV']

        # Plot parameters
        colors = ['r', 'c', 'b', 'm', 'y', 'g', 'orange', '#ae34eb', '#eb348f', '#000000']
        COLORS = {p_name: colors[i] for i, p_name in enumerate(known_methods)}
        markers = ["o", "^", "v", "<", ">", ",", "h", "x", "+"]
        MARKERS = {p_name: markers[i] for i, p_name in enumerate(known_methods)}
        linestyle = ["-", "-", "--", "-.", ":", "--", "-.", ":", "-"]
        LINESTYLE = {p_name: linestyle[i] for i, p_name in enumerate(known_methods)}

        plt.figure(figsize=set_size(WIDTH))
        for i, m in enumerate(methods):
            if m == 'optimalMV':
                plt.plot(epsilons, means[i], color=COLORS['oracleMV'], marker=MARKERS['oracleMV'], label='oracleMV')
                plt.fill_between(epsilons, means[i]-stds[i]/2, means[i]+stds[i]/2, color=COLORS['oracleMV'], alpha=0.3) 
            else:
                plt.plot(epsilons, means[i], color=COLORS[m], marker=MARKERS[m], label=m)
                plt.fill_between(epsilons, means[i]-stds[i]/2, means[i]+stds[i]/2, color=COLORS[m], alpha=0.3)
          
        plt.legend()
        #plt.title("Comparing the covariance estimation errors in presence of {} contamination".format(cont))
        plt.xlabel(r"Dataset contamination rate ($1-\delta$)")
        plt.ylabel("Operator error of covariance estimator")
        if folder is None:
            filename = "nsample={}_dim={}_erank={}_cont={}_intensity={}_cont-type={}".format(nsample, dim, erank,cont, intensity, cont_type)
            plt.savefig('./figures/' + folder + '/' + filename+'.pdf', format='pdf', bbox_inches='tight')
            
            plt.show()
        else:
            filename = "nsample={}_dim={}_erank={}_cont={}_intensity={}_cont-type={}".format(nsample, dim, erank,cont, intensity, cont_type)
            plt.savefig('./figures/' + folder + '/' + filename+'.pdf', format='pdf', bbox_inches='tight')
            plt.close()