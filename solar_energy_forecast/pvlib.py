import numpy as np
import pandas as pd

def get_relative_airmass(zenith, model='kastenyoung1989'):
    '''
    Calculate relative (not pressure-adjusted) airmass at sea level.

    Parameter ``model`` allows selection of different airmass models.

    Parameters
    ----------
    zenith : numeric
        Zenith angle of the sun. [degrees]

    model : string, default 'kastenyoung1989'
        Available models include the following:

        * 'simple' - secant(apparent zenith angle) -
          Note that this gives -Inf at zenith=90
        * 'kasten1966' - See [1]_ - requires apparent sun zenith
        * 'youngirvine1967' - See [2]_ - requires true sun zenith
        * 'kastenyoung1989' (default) - See [3]_ - requires apparent sun zenith
        * 'gueymard1993' - See [4]_, [5]_ - requires apparent sun zenith
        * 'young1994' - See [6]_ - requires true sun zenith
        * 'pickering2002' - See [7]_ - requires apparent sun zenith
        * 'gueymard2003' - See [8]_, [9]_ - requires apparent sun zenith

    Returns
    -------
    airmass_relative : numeric
        Relative airmass at sea level. Returns NaN values for any
        zenith angle greater than 90 degrees. [unitless]

    Notes
    -----
    Some models use apparent (refraction-adjusted) zenith angle while
    other models use true (not refraction-adjusted) zenith angle. Apparent
    zenith angles should be calculated at sea level.

    Comparison among several models is reported in [10]_.

    References
    ----------
    .. [1] Fritz Kasten, "A New Table and Approximation Formula for the
       Relative Optical Air Mass," CRREL (U.S. Army), Hanover, NH, USA,
       Technical Report 136, 1965.
       :doi:`11681/5671`

    .. [2] A. T. Young and W. M. Irvine, "Multicolor Photoelectric
       Photometry of the Brighter Planets. I. Program and Procedure,"
       The Astronomical Journal, vol. 72, pp. 945-950, 1967.
       :doi:`10.1086/110366`

    .. [3] Fritz Kasten and Andrew Young, "Revised optical air mass tables
       and approximation formula," Applied Optics 28:4735-4738, 1989.
       :doi:`10.1364/AO.28.004735`

    .. [4] C. Gueymard, "Critical analysis and performance assessment of
       clear sky solar irradiance models using theoretical and measured
       data," Solar Energy, vol. 51, pp. 121-138, 1993.
       :doi:`10.1016/0038-092X(93)90074-X`

    .. [5] C. Gueymard, "Development and performance assessment of a clear
       sky spectral radiation model,” in Proc. of the 22nd ASES Conference,
       Solar ’93, 1993, pp. 433–438.

    .. [6] A. T. Young, "Air-Mass and Refraction," Applied Optics, vol. 33,
       pp. 1108-1110, Feb. 1994.
       :doi:`10.1364/AO.33.001108`

    .. [7] Keith A. Pickering, "The Southern Limits of the Ancient Star Catalog
       and the Commentary of Hipparchos," DIO, vol. 12, pp. 3-27, Sept. 2002.
       Available at `DIO <http://dioi.org/jc01.pdf>`_

    .. [8] C. Gueymard, "Direct solar transmittance and irradiance
       predictions with broadband models. Part I: detailed theoretical
       performance assessment". Solar Energy, vol 74, pp. 355-379, 2003.
       :doi:`10.1016/S0038-092X(03)00195-6`

    .. [9] C. Gueymard, "Clear-Sky Radiation Models and Aerosol Effects", in
       Solar Resources Mapping: Fundamentals and Applications,
       Polo, J., Martín-Pomares, L., Sanfilippo, A. (Eds), Cham, CH: Springer,
       2019, pp. 137-182.
       :doi:`10.1007/978-3-319-97484-2_5`

    .. [10] Matthew J. Reno, Clifford W. Hansen and Joshua S. Stein, "Global
       Horizontal Irradiance Clear Sky Models: Implementation and Analysis"
       Sandia National Laboratories, Albuquerque, NM, USA, SAND2012-2389, 2012.
       :doi:`10.2172/1039404`

    '''

    # set zenith values greater than 90 to nans
    z = np.where(zenith > 90, np.nan, zenith)
    zenith_rad = np.radians(z)

    model = model.lower()

    if 'kastenyoung1989' == model:
        am = (1.0 / (np.cos(zenith_rad) +
              0.50572*((6.07995 + (90 - z)) ** - 1.6364)))
    elif 'kasten1966' == model:
        am = 1.0 / (np.cos(zenith_rad) + 0.15*((93.885 - z) ** - 1.253))
    elif 'simple' == model:
        am = 1.0 / np.cos(zenith_rad)
    elif 'pickering2002' == model:
        am = (1.0 / (np.sin(np.radians(90 - z +
              244.0 / (165 + 47.0 * (90 - z) ** 1.1)))))
    elif 'youngirvine1967' == model:
        sec_zen = 1.0 / np.cos(zenith_rad)
        am = sec_zen * (1 - 0.0012 * (sec_zen * sec_zen - 1))
    elif 'young1994' == model:
        am = ((1.002432*((np.cos(zenith_rad)) ** 2) +
              0.148386*(np.cos(zenith_rad)) + 0.0096467) /
              (np.cos(zenith_rad) ** 3 +
              0.149864*(np.cos(zenith_rad) ** 2) +
              0.0102963*(np.cos(zenith_rad)) + 0.000303978))
    elif 'gueymard1993' == model:  # [4], Eq. 22 and [5], Eq. 3b
        am = (1.0 / (np.cos(zenith_rad) +
              0.00176759*(z)*((94.37515 - z) ** - 1.21563)))
    elif 'gueymard2003' == model:
        am = (1.0 / (np.cos(zenith_rad) +
              0.48353*(z**0.095846)/(96.741 - z)**1.754))
    else:
        raise ValueError('%s is not a valid model for relativeairmass', model)

    if isinstance(zenith, pd.Series):
        am = pd.Series(am, index=zenith.index)

    return am