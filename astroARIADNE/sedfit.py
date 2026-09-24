from astroARIADNE.star import Star
from astroARIADNE.fitter import Fitter
from astroARIADNE.plotter import SEDPlotter


def main():
    starnames = ['HD27371','HD27697','HD28305','HD28307']
    ras = [64.94888616843,65.73422490903,67.15467370963,67.14423196264]
    decs = [15.62753819285,17.54239075752,19.18027331731,15.96202530801]
    gaia_ids = [3312052249215275904,3314024566919613952,48026706558487040,3312748824193704832]
    
    #When you run this the first time, when it compiles the photometry, astroARIADNE tells you which photometry points may be bad. You can remove them by adding them to the lists below
    removes = [
        ['GALEX_FUV','GALEX_NUV','STROMGREN_u','GROUND_JOHNSON_U','STROMGREN_b'],
        ['GALEX_FUV','STROMGREN_u','GROUND_JOHNSON_U','STROMGREN_v'],
        ['GALEX_FUV','STROMGREN_u','GROUND_JOHNSON_U','STROMGREN_v'],
        ['STROMGREN_u','GROUND_JOHNSON_U','STROMGREN_b'],
    ]
    
    prior_teff = [('default'),('default'),('default'),('default')]
    prior_logg = [('normal',3,0.26),('normal',2.97,0.29),('normal',3.12,0.26),('normal',3.21,0.28)]
    prior_z    = [('normal',0.11,0.1),('normal',0.1,0.12),('normal',0.15,0.11),('normal',0.19,0.11)]
    prior_dist = [('default'),('default'),('default'),('default')]
    prior_rad  = [('default'),('default'),('default'),('default')]
    prior_Av   = [('uniform',0.0,0.001),('uniform',0.0,0.001),('uniform',0.0,0.001),('uniform',0.0,0.001)]
    
    stars = []
    for i,starname in enumerate(starnames):
        print(starname)
        star = Star(starnames[i],ras[i],decs[i],g_id=gaia_ids[i])
        for remove in removes[i]:
            star.remove_mag(remove)
        star.prior = {
            'teff': prior_teff[i], 'logg': prior_logg[i], 'z': prior_z[i],
    	    'dist': prior_dist[i], 'rad': prior_rad[i], 'Av': prior_Av[i],
    }
        stars.append(star)
    
    for s in stars:
        run_sed(s)
    
def run_sed(s):
    out_folder = '{}_output'.format(s.starname)

    engine = 'dynesty'
    nlive = 500
    dlogz = 0.5
    bound = 'multi'
    sample = 'rwalk'
    threads = 4
    dynamic = False

    setup = [engine, nlive, dlogz, bound, sample, threads, dynamic]

    # Feel free to uncomment any unneeded/unwanted models
    models = [
	    'phoenix',
	    'btsettl',
	    'btnextgen',
	    'btcond',
	    'kurucz',
	    'ck04',
	    'bosz'      # MARCS+ATLAS9/Synspec, 2800-16000 K (BMA-capable)
    ]

    f = Fitter()
    f.star = s
    f.setup = setup
    f.av_law = 'fitzpatrick'
    f.out_folder = out_folder
    f.bma = True
    #f.norm = True

    #f.grid = 'phoenix'      # or 'bosz', 'sphinx', 'tlusty', 'coelho', ...
    f.models = models
    f.n_samples = 100000
    #f.prior_setup = {
	#    'teff': ('default'), 'logg': ('uniform',0.0,3.0), 'z': ('default'),
	#    'dist': ('default'), 'rad': ('default'), 'Av': ('uniform',0.0,0.001),
    #}
    f.prior_setup = s.prior
    f.estimate_age=False
    f.initialize()
    f.fit_bma()
    
    
if __name__ == '__main__':
    main()
