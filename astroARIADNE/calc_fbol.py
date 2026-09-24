import numpy as np
import pickle
import os
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde, norm
from scipy.special import erf

def main():
    starnames = ['HD27371','HD27697','HD28305','HD28307']
    
    for starname in starnames:
        print('Calculating Fbol for {}'.format(starname))
        do_fbol(starname)
        
def do_fbol(starname, out_dir=None):
    if out_dir is None:
        out_dir = '{}_output'.format(starname)
    fn = os.path.join(out_dir, 'a.{}.BMA.pkl'.format(starname))

    with open(fn,'rb') as f:
        data = pickle.load(f)


    ws = data['weighted_samples']
    wa = data['weighted_average']
    ws_fbol = get_fbol(ws['lum'],ws['dist'])
    wa_fbol = get_fbol(wa['lum'],wa['dist'])

    orig = data['originals']

    mod_fbols = dict()
    for model in orig:
        mod_fbol = get_fbol(orig[model]['lum'],orig[model]['dist'])
        mod_fbols[model] = mod_fbol

    full_report = ''
    full_report += 'Weighted Samples\n'
    ws_report = fbol_report(ws_fbol)
    full_report += ws_report
    full_report += '\nWeighted Average\n'
    wa_report = fbol_report(wa_fbol)
    full_report += wa_report

    for model in mod_fbols:
        full_report += '\n{}\n'.format(model)
        mod_report = fbol_report(mod_fbols[model])
        full_report += mod_report

    #write report
    report_out = os.path.join(out_dir, 'a.{}.fbol_report.txt'.format(starname))
    with open(report_out, 'w') as f:
        f.write(full_report)

    #Do plotting
    plot_dir = os.path.join(out_dir,'plots','histograms')
    os.makedirs(plot_dir, exist_ok=True)
    fbol_plot(plot_dir,starname,data,ws_fbol,wa_fbol,mod_fbols)
    
def fbol_report(samp):
    #Report the statistics on the fbol
    #(based on out_filler in utils.py)
    xx,pdf = estimate_pdf(samp)
    cdf = estimate_cdf(samp, hdr=True)
    best, lo, up = credibility_interval_hdr(xx, pdf, cdf, sigma=1)
    _, lo3, up3 = credibility_interval_hdr(xx, pdf, cdf, sigma=3)
    
    report = ''
    
    report += 'Best fit: {} erg/s/cm2 \n'.format(best)
    report += 'lo/hi 1-sig: {} - {} erg/s/cm2 \n'.format(lo,up)
    report += '    %: {} - {} \n'.format(lo/best*100,up/best*100)
    report += 'lo/hi 3-sig: {} - {} erg/s/cm2 \n'.format(lo3,up3)
    report += '    %: {} - {} \n'.format(lo3/best*100,up3/best*100)
    
    return report
    
    
def fbol_plot(plot_dir,starname,data,ws_fbol,wa_fbol,mod_fbols):
    #Here, I take the plotting method from plotter, plot_bma_hist to make sure we're getting the same flavor of plot for the flux
        
    colors = [
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
        'tab:brown'
    ]
    f1, ax1 = plt.subplots(figsize=(4.5, 3.25), dpi=300)
    f2, ax2 = plt.subplots(figsize=(4.5, 3.25), dpi=300)
    #Plot fbol histogram of the different model fits
    for j, m in enumerate(mod_fbols):
        # Get samples
        samp = mod_fbols[m]
        #samp = orig[m]['teff']
        # Plot sample histogram
        label = m + ' prob: {:.3f}'.format(data['weights'][m])
        # Normal
        n, bins1, patches = ax1.hist(samp, alpha=.3, bins=20,
                                     label=label, density=True,
                                     color=colors[j])
        # Weighted
        n, bins2, patches = ax2.hist(
            samp, alpha=.3, bins=20, label=label,
            weights=[data['weights'][m]] * len(samp)
        )
        # Fit a KDE to data
        kde = gaussian_kde(samp)
        # Estimate amplitude of the weighted distributions
        mu, sig = norm.fit(samp)
        try:
            bc = bins2[:-1] + np.diff(bins2)
            popt, pcov = curve_fit(norm_fit, xdata=bc, ydata=n,
                                   p0=[mu, sig, n.max()],
                                   maxfev=50000)
        except:
            popt = (mu, sig, n.max())
            print('exception')
        xx1 = np.linspace(bins1[0], bins1[-1], 1000)
        xx2 = np.linspace(bins2[0], bins2[-1], 1000)
        # Plot best fit
        ax1.plot(xx1, kde(xx1), lw=2, alpha=1, color=colors[j])
        ax2.plot(xx2, kde(xx2) * popt[2], lw=2, alpha=1,
                 color=colors[j])
    # The same but for the weighted samples
    n, bins, patches = ax1.hist(
        ws_fbol, alpha=.3,
        bins=20, label='Weighted sampling', density=True,
        color='tab:cyan'
    )
    kde = gaussian_kde(ws_fbol)
    xx = np.linspace(bins[0], bins[-1], 300)
    ax1.plot(xx, kde(xx), color='tab:cyan', lw=2, alpha=1, ls='--')

    # Now ditto for the weighted average
    n, bins, patches = ax1.hist(
        wa_fbol, alpha=.3,
        bins=20, label='Weighted average', density=True,
        color='tab:pink'
    )
    kde = gaussian_kde(wa_fbol)
    xx = np.linspace(bins[0], bins[-1], 300)
    ax1.plot(xx, kde(xx), color='tab:pink', lw=2, alpha=1, ls='-.')
    
    lab = r'F_bol (erg/s/cm^2)'
    
    # Normal
    ax1.set_ylabel('PDF', fontsize=10)
    # Weighted
    ax2.set_ylabel('N', fontsize=10)
    axes = [ax1, ax2]
    for ax in axes:
        ax.set_xlabel(lab, fontsize=10)

        ax.tick_params(
            axis='both', which='major', labelsize=12
        )
        ax.legend(loc=0, prop={'size': 9})
    
    #ax1.set_ylim([0,0.006])
    #ax2.set_ylim([0,1200])
    #ax1.set_xlim([2e-40,1.6e-39])
    #ax2.set_xlim([2e-40,1.6e-39])
    
    f1.savefig(os.path.join(plot_dir,'a.{}.fbol.png'.format(starname)),
               bbox_inches='tight')
    f2.savefig(os.path.join(plot_dir,'a.{}.weighted_fbol.png'.format(starname)),
               bbox_inches='tight')
    plt.close(f1)
    plt.close(f2)

def get_fbol(lum,dist):
    '''Takes a lum and dist array and calculates the fbol array from them'''
    #convert pc to cm
    this_dist = dist * 3.0857e18
    #conver L_sun to erg/s
    this_lum = lum * 3.828e33
    fbol = this_lum/(4*np.pi*this_dist**2)
    return fbol



def estimate_pdf(distribution):
    """Estimates the PDF of a distribution using a gaussian KDE. Taken from astroARIADNE utils.py.

    Parameters
    ----------
    distribution: array_like
        The distribution.
    Returns
    -------
    xx: array_like
        The x values of the PDF.
    pdf: array_like
        The estimated PDF.
    """
    kde = gaussian_kde(distribution)
    xmin, xmax = distribution.min(), distribution.max()
    xx = np.linspace(xmin, xmax, 300)
    pdf = kde(xx)
    return xx, pdf


def estimate_cdf(distribution, hdr=False):
    """Estimate the CDF of a distribution. Taken from astroARIADNE utils.py."""
    h, hx = np.histogram(distribution, density=True, bins=499)
    cdf = np.zeros(500)  # ensure the first value of the CDF is 0
    if hdr:
        idx = np.argsort(h)[::-1]
        cdf[1:] = np.cumsum(h[idx]) * np.diff(hx)
    else:
        cdf[1:] = np.cumsum(h) * np.diff(hx)
    return cdf


def norm_fit(x, mu, sigma, A):
    """Gaussian function. Taken from astroARIADNE utils.py."""
    return A * norm.pdf(x, loc=mu, scale=sigma)

def credibility_interval_hdr(xx, pdf, cdf, sigma=1.):
    """Calculate the highest density region for an empirical distribution. Taken from astroARIADNE utils.py.

    Reference: Hyndman, Rob J. 1996

    Parameters
    ----------
    xx: array_like
        The x values of the PDF (and the y values of the CDF).
    pdf: array_like
        The PDF of the distribution.
    cdf: array_like
        The CDF of the distribution.
    sigma: float
        The confidence level in sigma notation. (e.g. 1 sigma = 68%)

    Returns
    -------
    best: float
        The value corresponding to the peak of the posterior distribution.
    low: float
        The minimum value of the HDR.
    high: float
        The maximum value of the HDR.

    Note: The HDR is capable of calculating more robust credible regions
    for multimodal distributions. It is identical to the usual probability
    regions of symmetric about the mean distributions. Using this then should
    lead to more realistic errorbars and 3-sigma intervals for multimodal
    outputs.

    """
    # Get best fit value
    best = xx[np.argmax(pdf)]
    z = erf(sigma / np.sqrt(2))
    # Sort the pdf in reverse order
    idx = np.argsort(pdf)[::-1]
    # Find where the CDF reaches 100*z%
    idx_hdr = np.where(cdf >= z)[0][0]
    # Isolate the HDR
    hdr = pdf[idx][:idx_hdr]
    # Get the minimum density
    hdr_min = hdr.min()
    # Get CI
    low = xx[pdf > hdr_min].min()
    high = xx[pdf > hdr_min].max()
    return best, low, high


if __name__ == '__main__':
    main()
    
