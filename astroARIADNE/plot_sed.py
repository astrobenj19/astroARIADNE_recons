from astroARIADNE.star import Star
from astroARIADNE.fitter import Fitter
from astroARIADNE.plotter import SEDPlotter


def main():
    starnames = ['HD27371','HD27697','HD28305','HD28307']
    
    for starname in starnames:
        make_plots(starname)

def make_plots(starname):
    out_folder = '{}_output'.format(starname)


    in_file = out_folder + '/BMA.pkl'
    plots_out_folder = out_folder+'/plots'

    artist = SEDPlotter(in_file, plots_out_folder)
    artist.plot_SED_no_model()
    artist.plot_SED()
    artist.plot_bma_hist()
    artist.plot_bma_HR(10)
    artist.plot_corner()

if __name__ == '__main__':
    main()
