"""
display_labels.py
-----------------
Matplotlib display labels and physical units for all features and targets
used in figure annotations across the pipeline.

    FEATURE_DISPLAY  — topographic features written by 02_extract_features.py
    TARGET_DISPLAY   — LE parameter targets used by 03_train_models.py and
                       04_test_hypotheses.py

Column names in FEATURE_DISPLAY must match those written into
features-{job_id}.pkl by 02_extract_features.py.  If a feature column is
renamed there, update the corresponding key here.

Each entry is a dict with two keys:
    'label'  — LaTeX string for axis labels and annotations
    'unit'   — LaTeX string for the physical unit, or '' if dimensionless
"""

# =============================================================================
# Feature display metadata
# Column order follows the group structure in 02_extract_features.py.
# =============================================================================

FEATURE_DISPLAY = {

    # --- Scaling features ----------------------------------------------------
    'Ly':       {'label': r'$L_y$',             'unit': r'$\mathrm{m}$'},
    'Z_max':    {'label': r'$z_\mathrm{max}$',  'unit': r'$\mathrm{m}$'},
    'n_nodes':  {'label': r'$N_\mathrm{nodes}$','unit': ''},

    # --- Elevation statistics ------------------------------------------------
    'Z_mean':   {'label': r'$\bar{z}$',         'unit': r'$\mathrm{m}$'},
    'Z_cv':     {'label': r'$C_v(z)$',          'unit': ''},
    'Z_med':    {'label': r'$z_\mathrm{med}$',  'unit': r'$\mathrm{m}$'},
    'Z_skew':   {'label': r'$\gamma_z$',         'unit': ''},
    'Z_kurt':   {'label': r'$\kappa_z$',         'unit': ''},

    # --- Gradient statistics -------------------------------------------------
    'grd_mean': {'label': r'$\overline{|\nabla z|}$',          'unit': ''},
    'grd_std':  {'label': r'$\sigma_{|\nabla z|}$',            'unit': ''},
    'grd_med':  {'label': r'$|\nabla z|_\mathrm{med}$',        'unit': ''},
    'grd_max':  {'label': r'$|\nabla z|_\mathrm{max}$',        'unit': ''},
    'grd_skew': {'label': r'$\gamma_{|\nabla z|}$',             'unit': ''},
    'grd_kurt': {'label': r'$\kappa_{|\nabla z|}$',             'unit': ''},

    # --- Curvature statistics ------------------------------------------------
    'crv_mean': {'label': r'$\overline{\nabla^2 z}$',          'unit': r'$\mathrm{m}^{-1}$'},
    'crv_std':  {'label': r'$\sigma_{\nabla^2 z}$',            'unit': r'$\mathrm{m}^{-1}$'},
    'crv_min':  {'label': r'$\nabla^2 z_\mathrm{min}$',        'unit': r'$\mathrm{m}^{-1}$'},
    'crv_med':  {'label': r'$\nabla^2 z_\mathrm{med}$',        'unit': r'$\mathrm{m}^{-1}$'},
    'crv_max':  {'label': r'$\nabla^2 z_\mathrm{max}$',        'unit': r'$\mathrm{m}^{-1}$'},
    'crv_skew': {'label': r'$\gamma_{\nabla^2 z}$',             'unit': ''},
    'crv_kurt': {'label': r'$\kappa_{\nabla^2 z}$',             'unit': ''},

    # --- Hilltop curvature statistics ----------------------------------------
    'htcrv_mean': {'label': r'$\overline{\nabla^2 z_\mathrm{ht}}$',    'unit': r'$\mathrm{m}^{-1}$'},
    'htcrv_std':  {'label': r'$\sigma_{\nabla^2 z_\mathrm{ht}}$',      'unit': r'$\mathrm{m}^{-1}$'},
    'htcrv_min':  {'label': r'$\nabla^2 z_{\mathrm{ht,min}}$',         'unit': r'$\mathrm{m}^{-1}$'},
    'htcrv_med':  {'label': r'$\nabla^2 z_{\mathrm{ht,med}}$',         'unit': r'$\mathrm{m}^{-1}$'},
    'htcrv_max':  {'label': r'$\nabla^2 z_{\mathrm{ht,max}}$',         'unit': r'$\mathrm{m}^{-1}$'},
    'htcrv_skew': {'label': r'$\gamma_{\nabla^2 z_\mathrm{ht}}$',       'unit': ''},
    'htcrv_kurt': {'label': r'$\kappa_{\nabla^2 z_\mathrm{ht}}$',       'unit': ''},

    # --- Hypsometry ----------------------------------------------------------
    'hyp_int':  {'label': r'$\mathcal{H}$',     'unit': ''},

    # --- Network: whole-network ratios ---------------------------------------
    'Rb':       {'label': r'$\overline{R_b}$',  'unit': ''},
    'Rl':       {'label': r'$\overline{R_l}$',  'unit': ''},

    # --- Network: zero-order ratios ------------------------------------------
    'Rb0':      {'label': r'$R_{b0}$',          'unit': ''},
    'Rl0':      {'label': r'$R_{l0}$',          'unit': ''},

    # --- Network: zero-order channel geometry --------------------------------
    'n0':       {'label': r'$N_0$',             'unit': ''},
    'L0':       {'label': r'$L_0$',             'unit': r'$\mathrm{m}$'},
    'l0_mean':  {'label': r'$\bar{l}_0$',       'unit': r'$\mathrm{m}$'},
    'rlf0_mean':{'label': r'$\overline{\Delta z_0}$', 'unit': r'$\mathrm{m}$'},
    'grd0_mean':{'label': r'$\overline{S_0}$',  'unit': ''},

    # --- Network: path length ------------------------------------------------
    'path_max': {'label': r'$P_\mathrm{max}$',  'unit': r'$\mathrm{m}$'},
}


# =============================================================================
# Target display metadata
# =============================================================================

TARGET_DISPLAY = {
    # Individual LE parameters
    'u':         {'label': r'$U$',           'unit': r'$\mathrm{m\,yr^{-1}}$'},
    'kh':        {'label': r'$K_h$',         'unit': r'$\mathrm{m^2\,yr^{-1}}$'},
    'ks':        {'label': r'$K_s$',         'unit': r'$\mathrm{yr^{-1}}$'},

    # Dimensionless parameter ratios (stored as raw values in features-*.pkl;
    # log10-transformed at training time in 03_train_models.py)
    'u_ks':  {'label': r'$U/K_s$',       'unit': r'$\mathrm{m}$'},
    'kh_ks': {'label': r'$K_h/K_s$',     'unit': r'$\mathrm{m^2}$'},
    'u_kh':  {'label': r'$U/K_h$',       'unit': r'$\mathrm{m^{-1}}$'},
}


# =============================================================================
# Convenience helper
# =============================================================================

def axis_label(col: str, log: bool = False) -> str:
    """Return a formatted axis label string for *col*.

    Parameters
    ----------
    col : str
        Column name — must be a key in FEATURE_DISPLAY or TARGET_DISPLAY.
    log : bool
        If True, wrap the symbol in ``log_{10}(...)``.

    Returns
    -------
    str
        E.g. ``r'$\\log_{10}(U/K_s)\\;[\\mathrm{m}]$'``
    """
    lookup = {**FEATURE_DISPLAY, **TARGET_DISPLAY}
    if col not in lookup:
        return col
    entry = lookup[col]
    sym  = entry['label']
    unit = entry['unit']

    inner = rf'\log_{{10}}({sym.strip("$")})' if log else sym.strip('$')
    label = rf'${inner}$'
    if unit:
        label += rf'$\;[{unit.strip("$")}]$'
    return label
