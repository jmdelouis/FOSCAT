import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import math
import io
import requests
from PIL import Image

def lgnomproject(
        cell_ids,               # (N,) HEALPix pixel indices
        data,                   # (N,) scalar values OR (N,3) RGB values in [0,1]
        nside: int,
        rot=None,
        xsize: int = 400,
        ysize: int = 400,
        reso: float = None,
        fov_deg=None,
        nest: bool = True,
        reduce: str = "mean",   # 'mean'|'median'|'sum'|'first'
        mask_outside: bool = True,
        unseen_value=None,      # defaults to hp.UNSEEN (scalar) or np.nan (RGB)
        return_image_only: bool = False,
        title: str = None, cmap: str = "viridis", vmin=None, vmax=None,
        notext: bool = False,
        hold: bool = True,
        interp: bool = False,
        sub=(1,1,1),
        cbar: bool = False,
        unit: str = "Value",
        rgb_clip=(0.0, 1.0),    # clip range for RGB
):
    """
    Gnomonic projection from *sparse* HEALPix samples (cell_ids, data) to an image.
    Supports scalar data (N,) and RGB data (N,3). For RGB, colorbar/cmap/vmin/vmax are ignored.
    """

    # -------- 0) Input normalization --------
    if unseen_value is None:
        unseen_value = hp.UNSEEN if (np.ndim(data) == 1 or (np.ndim(data)==2 and data.shape[1] != 3)) else np.nan

    cell_ids = np.asarray(cell_ids, dtype=np.int64)
    vals_in  = np.asarray(data)

    if vals_in.ndim == 1:
        is_rgb = False
        if cell_ids.shape != vals_in.shape:
            raise ValueError("For scalar mode, `data` must have shape (N,).")
        vals_in = vals_in.astype(float)
    elif vals_in.ndim == 2 and vals_in.shape[1] == 3:
        is_rgb = True
        if cell_ids.shape[0] != vals_in.shape[0]:
            raise ValueError("For RGB mode, `data` must have shape (N,3) matching `cell_ids` length.")
        vals_in = vals_in.astype(float)
    else:
        raise ValueError("`data` must be (N,) for scalar or (N,3) for RGB.")

    # -------- 1) Aggregate duplicates in cell_ids --------
    uniq, inv = np.unique(cell_ids, return_inverse=True)  # uniq sorted
    if is_rgb:
        if reduce == "first":
            first_idx = np.full(uniq.size, -1, dtype=np.int64)
            for i, g in enumerate(inv):
                if first_idx[g] < 0:
                    first_idx[g] = i
            agg_vals = vals_in[first_idx, :]  # (U,3)
        elif reduce == "sum":
            agg_vals = np.zeros((uniq.size, 3), dtype=float)
            # sum per channel
            for c in range(3):
                np.add.at(agg_vals[:, c], inv, vals_in[:, c])
        elif reduce == "median":
            agg_vals = np.empty((uniq.size, 3), dtype=float)
            for k, pix in enumerate(uniq):
                sel = (cell_ids == pix)
                agg_vals[k, :] = np.median(vals_in[sel, :], axis=0)
        elif reduce == "mean":
            sums = np.zeros((uniq.size, 3), dtype=float)
            cnts = np.zeros(uniq.size, dtype=float)
            for c in range(3):
                np.add.at(sums[:, c], inv, vals_in[:, c])
            np.add.at(cnts, inv, 1.0)
            agg_vals = sums / np.maximum(cnts[:, None], 1.0)
        else:
            raise ValueError("reduce must be one of {'mean','median','sum','first'}")
    else:
        # scalar path (comme ta version)
        if reduce == "first":
            first_idx = np.full(uniq.size, -1, dtype=np.int64)
            for i, g in enumerate(inv):
                if first_idx[g] < 0:
                    first_idx[g] = i
            agg_vals = vals_in[first_idx]
        elif reduce == "sum":
            agg_vals = np.zeros(uniq.size, dtype=float)
            np.add.at(agg_vals, inv, vals_in)
        elif reduce == "median":
            agg_vals = np.empty(uniq.size, dtype=float)
            for k, pix in enumerate(uniq):
                agg_vals[k] = np.median(vals_in[cell_ids == pix])
        elif reduce == "mean":
            sums = np.zeros(uniq.size, dtype=float)
            cnts = np.zeros(uniq.size, dtype=float)
            np.add.at(sums, inv, vals_in)
            np.add.at(cnts, inv, 1.0)
            agg_vals = sums / np.maximum(cnts, 1.0)
        else:
            raise ValueError("reduce must be one of {'mean','median','sum','first'}")

    # -------- 2) Choose gnomonic center (rot) --------
    if rot is None:
        theta_c, phi_c = hp.pix2ang(nside, uniq, nest=nest)  # colat, lon (rad)
        lon0_deg = np.degrees(np.angle(np.mean(np.exp(1j * phi_c))))
        lat0_deg = 90.0 - np.degrees(np.median(theta_c))
        psi_deg  = 0.0
        rot = (lon0_deg % 360.0, float(lat0_deg), float(psi_deg))

    lon0_deg, lat0_deg, psi_deg = rot
    lon0 = np.deg2rad(lon0_deg)
    lat0 = np.deg2rad(lat0_deg)
    psi  = np.deg2rad(psi_deg)

    # -------- 3) Tangent-plane grid --------
    if reso is not None:
        dx = np.tan(np.deg2rad(reso))
        dy = dx
        half_x = 0.5 * xsize * dx
        half_y = 0.5 * ysize * dy
    else:
        if fov_deg is None:
            fov_deg = np.rad2deg(np.sqrt(cell_ids.shape[0]) / nside) * 1.4
        if np.isscalar(fov_deg):
            fx, fy = float(fov_deg), float(fov_deg)
        else:
            fx, fy = float(fov_deg[0]), float(fov_deg[1])
        ax = np.deg2rad(0.5 * fx)
        ay = np.deg2rad(0.5 * fy)
        half_x = np.tan(ax)
        half_y = np.tan(ay)

    xs = np.linspace(-half_x, +half_x, xsize, endpoint=False) + (half_x / xsize)
    ys = np.linspace(-half_y, +half_y, ysize, endpoint=False) + (half_y / ysize)
    X, Y = np.meshgrid(xs, ys)

    # rotate plane
    c, s = np.cos(psi), np.sin(psi)
    Xr =  c * X + s * Y
    Yr = -s * X + c * Y

    # -------- 4) Inverse gnomonic → sphere --------
    rho  = np.hypot(Xr, Yr)
    cang = np.arctan(rho)
    sinc, cosc = np.sin(cang), np.cos(cang)
    sinlat0, coslat0 = np.sin(lat0), np.cos(lat0)

    with np.errstate(invalid="ignore", divide="ignore"):
        lat = np.arcsin(cosc * sinlat0 + (Yr * sinc * coslat0) / np.where(rho == 0, 1.0, rho))
        lon = lon0 + np.arctan2(Xr * sinc, rho * coslat0 * cosc - Yr * sinlat0 * sinc)

    lon = (lon + 2*np.pi) % (2*np.pi)
    theta_img = (np.pi / 2.0) - lat
    outside = (cosc <= 0.0) if mask_outside else np.zeros_like(cosc, dtype=bool)

    if interp:

        # Inputs attendus (déjà présents dans ton code):
        # - nside, nest
        # - theta_img, lon  (angles pour chaque pixel de l'image, shape (ysize, xsize))
        # - uniq (np.ndarray trié d'indices HEALPix présents)
        # - agg_vals : valeurs associées à uniq
        #     * si is_rgb: shape (uniq.size, 3)
        #     * sinon:     shape (uniq.size,)
        # - ysize, xsize
        # - mask_outside (bool)
        # - outside (masque bool à plat ou 2D selon ton code)
        # - unseen_value (float), p.ex. np.nan ou autre

        # -------- 5) (NOUVEAU) Interpolation bilinéaire via poids HEALPix --------
        # Aplatis les angles de l'image
        theta_flat = theta_img.ravel()
        phi_flat   = lon.ravel()

        # Récupère pour chaque direction les indices de 4 voisins et leurs poids
        # inds: shape (npix_img, 4) ; w: shape (npix_img, 4)
        inds, w = hp.get_interp_weights(nside, theta_flat, phi_flat, nest=nest)

        # On mappe les indices 'inds' (HEALPix) vers positions dans 'uniq' en O(log N)
        pos = np.searchsorted(uniq, inds, side="left")
        in_range = pos < uniq.size
        match = np.zeros_like(in_range, dtype=bool)
        match[in_range] = (uniq[pos[in_range]] == inds[in_range])

        # Construit un masque 'valid' des voisins présents dans tes données
        # valid shape (npix_img, 4)
        valid = match

        if is_rgb:
            # Récupère les valeurs des 4 voisins (3 canaux)
            # vals shape (npix_img, 4, 3) avec NaN pour voisins absents
            vals = np.full((inds.shape[0], inds.shape[1], 3), np.nan, dtype=float)
            # positions valides -> on insère les vraies valeurs RGB
            vals[valid, :] = agg_vals[pos[valid], :]

            # pondération : on annule le poids là où la valeur est absente
            w_eff = w.copy()
            w_eff[~valid] = 0.0
            ws = np.sum(w_eff, axis=0, keepdims=False)  # somme des poids utiles

            # éviter division par 0 : pixels hors couverture -> unseen
            nonzero = ws.squeeze() > 0
            img_flat = np.full((inds.shape[1], 3), unseen_value, dtype=float)

            # combinaison pondérée (en ignorant les NaN via w_eff)
            num = np.nansum(vals * w_eff[..., None], axis=0)  # (npix_img, 3)
            img_flat[nonzero, :] = (num[nonzero, :] / ws[nonzero,None]).astype(float)

            img = img_flat.reshape(ysize, xsize, 3)

            if mask_outside:
                mask = outside.reshape(ysize, xsize)
                img[mask, :] = np.nan  # ou unseen_value

        else:
            # Scalaire : vals shape (npix_img, 4)
            vals = np.full(inds.shape, np.nan, dtype=float)
            vals[valid] = agg_vals[pos[valid]]

            w_eff = w.copy()
            w_eff[~valid] = 0.0
            ws = np.sum(w_eff, axis=0)  # (,npix_img)

            img_flat = np.full(inds.shape[1], unseen_value, dtype=float)
            nonzero = ws > 0

            num = np.nansum(vals * w_eff, axis=0)  # (npix_img,)
            img_flat[nonzero] = (num[nonzero] / ws[nonzero]).astype(float)

            img = img_flat.reshape(ysize, xsize)
            if mask_outside:
                img[outside] = unseen_value

    else:
        # -------- 5) Map image pixels to HEALPix ids --------
        ip_img = hp.ang2pix(nside, theta_img.ravel(), lon.ravel(), nest=nest).astype(np.int64)

        # -------- 6) Assign values by matching ip_img ∈ uniq --------
        pos = np.searchsorted(uniq, ip_img, side="left")
        valid = pos < uniq.size
        match = np.zeros_like(valid, dtype=bool)
        match[valid] = (uniq[pos[valid]] == ip_img[valid])
        
        if is_rgb:
            img_flat = np.full((ip_img.size, 3), np.nan, dtype=float)
            img_flat[match, :] = agg_vals[pos[match], :]
            img = img_flat.reshape(ysize, xsize, 3)
            if mask_outside:
                mask = outside.reshape(ysize, xsize)
                img[mask, :] = np.nan
        else:
            img_flat = np.full(ip_img.shape, unseen_value, dtype=float)
            img_flat[match] = agg_vals[pos[match]]
            img = img_flat.reshape(ysize, xsize)
            if mask_outside:
                img[outside] = unseen_value

    # -------- 7) Return / plot --------
    if return_image_only:
        return img

    # axes extents (approx)
    x_deg = np.degrees(np.arctan(xs))
    y_deg = np.degrees(np.arctan(ys))
    longitude_min = x_deg[0]/np.cos(np.deg2rad(lat0_deg)) + lon0_deg
    longitude_max = x_deg[-1]/np.cos(np.deg2rad(lat0_deg)) + lon0_deg
    if longitude_min > 180:
        longitude_min -= 360
        longitude_max -= 360
    extent = (longitude_min, longitude_max, y_deg[0]+lat0_deg, y_deg[-1]+lat0_deg)

    if hold:
        fig, ax = plt.subplots(figsize=(xsize/100, ysize/100), dpi=100)
    else:
        ax = plt.subplot(sub[0], sub[1], sub[2])

    if is_rgb:
        shown = ax.imshow(
            np.clip(img, rgb_clip[0], rgb_clip[1]),
            origin="lower",
            extent=extent,
            interpolation="nearest",
            aspect="auto"
        )
        # pas de cmap/cbar en RGB
    else:
        shown = ax.imshow(
            np.where(img == unseen_value, np.nan, img),
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            interpolation="nearest",
            aspect="auto"
        )
        if cbar:
            if hold:
                cb = fig.colorbar(shown, ax=ax)
                cb.set_label("value")
            else:
                plt.colorbar(shown, ax=ax, orientation="horizontal", label=unit)

    if not notext:
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return (fig, ax) if hold else ax

def plot_scat(s1,s2,s3,s4):

    if not isinstance(s1,np.ndarray): # manage if Torch tensor
        S1=s1.cpu().numpy()
        S2=s2.cpu().numpy()
        S3=s3.cpu().numpy()
        S4=s4.cpu().numpy()
    else:
        S1=s1
        S2=s2
        S3=s3
        S4=s4
        
    N_image=s1.shape[0]
    J=s1.shape[1]
    N_orient=s1.shape[2]

    # compute index j1 and j2 for S3
    j1_s3=np.zeros([s3.shape[1]],dtype='int')
    j2_s3=np.zeros([s3.shape[1]],dtype='int')
    
    # compute index j1 and j2 for S4
    j1_s4=np.zeros([s4.shape[1]],dtype='int')
    j2_s4=np.zeros([s4.shape[1]],dtype='int')
    j3_s4=np.zeros([s4.shape[1]],dtype='int')

    n_s3=0
    n_s4=0
    for j3 in range(0,J):
        for j2 in range(0, j3 + 1):
            j1_s3[n_s3]=j2
            j2_s3[n_s3]=j3
            n_s3+=1
            for j1 in range(0, j2 + 1):
                j1_s4[n_s4]=j1
                j2_s4[n_s4]=j2
                j3_s4[n_s4]=j3
                n_s4+=1


    color=['b','r','orange','pink']
    symbol=['',':','-','.']
    plt.figure(figsize=(16,12))
    
    plt.subplot(2,2,1)
    for k in range(4):
        plt.plot(S1[0,:,k],color=color[k%len(color)],label=r'$\Theta = %d$'%(k))
    plt.legend(frameon=0,ncol=2)
    plt.xlabel(r'$J_1$')
    plt.ylabel(r'$S_1$')
    plt.yscale('log')
    
    plt.subplot(2,2,2)
    for k in range(4):
        plt.plot(S2[0,:,k],color=color[k%len(color)],label=r'$\Theta = %d$'%(k))
    plt.xlabel(r'$J_1$')
    plt.ylabel(r'$S_2$')
    plt.yscale('log')
    
    plt.subplot(2,2,3)
    nidx=np.concatenate([np.zeros([1]),np.cumsum(np.bincount(j1_s3))],0)
    l_pos=[]
    l_name=[]
    for i in np.unique(j1_s3):
        idx=np.where(j1_s3==i)[0]
        for k in range(4):
            for l in range(4):
                if i==0:
                    plt.plot(j2_s3[idx]+nidx[i],S3[0,idx,k,l],symbol[l%len(symbol)],color=color[k%len(color)],label=r'$\Theta = %d,%d$'%(k,l))
                else:
                    plt.plot(j2_s3[idx]+nidx[i],S3[0,idx,k,l],symbol[l%len(symbol)],color=color[k%len(color)])
        l_pos=l_pos+list(j2_s3[idx]+nidx[i])
        l_name=l_name+["%d,%d"%(j1_s3[m],j2_s3[m]) for m in idx]
    plt.legend(frameon=0,ncol=2)
    
    plt.xticks(l_pos,l_name, fontsize=6)
    plt.xlabel(r"$j_{1},j_{2}$", fontsize=9)
    plt.ylabel(r"$S_{3}$", fontsize=9)
    
    plt.subplot(2,2,4)
    nidx=0
    l_pos=[]
    l_name=[]
    for i in np.unique(j1_s4):
        for j in np.unique(j2_s4):
            idx=np.where((j1_s4==i)*(j2_s4==j))[0]
            for k in range(4):
                for l in range(4):
                    for m in range(4):
                        if i==0 and j==0 and m==0:
                            plt.plot(j2_s4[idx]+j3_s4[idx]+nidx,S4[0,idx,k,l,m],symbol[l%len(symbol)],color=color[k%len(color)],label=r'$\Theta = %d,%d,%d$'%(k,l,m))
                        else:
                            plt.plot(j2_s4[idx]+j3_s4[idx]+nidx,S4[0,idx,k,l,m],symbol[l%len(symbol)],color=color[k%len(color)])
            l_pos=l_pos+list(j2_s4[idx]+j3_s4[idx]+nidx)
            l_name=l_name+["%d,%d,%d"%(j1_s4[m],j2_s4[m],j3_s4[m]) for m in idx]
        nidx+=np.max(j2_s4[j1_s4==i]+j3_s4[j1_s4==i])-np.min(j2_s4[j1_s4==i]+j3_s4[j1_s4==i])+1
    plt.legend(frameon=0,ncol=2)
    
    plt.xticks(l_pos,l_name, fontsize=6, rotation=90)
    plt.xlabel(r"$j_{1},j_{2},j_{3}$", fontsize=9)
    plt.ylabel(r"$S_{4}$", fontsize=9)


import numpy as np

def power_spectrum_1d(data, dx=1.0):
    """
    Compute the isotropic 1D power spectrum of a 2D field.

    Parameters
    ----------
    data : ndarray (ny, nx)
        Input 2D field.
    dx : float
        Pixel size in the same spatial unit as desired frequency inverse.
        If dx is in meters, returned frequencies are in m^-1 (cycles per meter).

    Returns
    -------
    f_centers : ndarray
        Radial spatial frequencies (cycles per unit length), e.g., m^-1 if dx is in meters.
    Pk : ndarray
        Azimuthally averaged power spectrum over radial frequency bins (arbitrary units unless you add a normalization).
    """
    # 2D FFT and power
    F = np.fft.fftshift(np.fft.fft2(data))
    P2D = np.abs(F) ** 2

    # Spatial frequency grids (cycles per unit length; NOT radians)
    ny, nx = data.shape
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))  # cycles per unit length (e.g., m^-1)
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dx))  # cycles per unit length (e.g., m^-1)
    fx2d, fy2d = np.meshgrid(fx, fy, indexing="xy")
    fr = np.sqrt(fx2d**2 + fy2d**2)  # radial spatial frequency (cycles per unit length)

    # Radial binning
    nbins = min(nx, ny) // 2
    f_bins = np.linspace(0.0, fr.max(), nbins + 1)

    # Vectorized bin average of P2D over annuli
    fr_flat = fr.ravel()
    P_flat = P2D.ravel()
    bin_idx = np.digitize(fr_flat, f_bins) - 1  # -> [0, nbins-1]
    valid = (bin_idx >= 0) & (bin_idx < nbins)

    # Sum and count per bin, then mean
    sum_per_bin = np.bincount(bin_idx[valid], weights=P_flat[valid], minlength=nbins)
    cnt_per_bin = np.bincount(bin_idx[valid], minlength=nbins)
    with np.errstate(invalid="ignore", divide="ignore"):
        Pk = sum_per_bin / cnt_per_bin
    Pk[cnt_per_bin == 0] = np.nan  # empty bins

    # Bin centers
    f_centers = 0.5 * (f_bins[1:] + f_bins[:-1])

    return f_centers, Pk

import numpy as np

def _freq_grids(ny, nx, dx=1.0):
    """Return 2D radial spatial frequency grid fr (cycles per unit), with fftshift."""
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dx))
    fx2d, fy2d = np.meshgrid(fx, fy, indexing="xy")
    fr = np.sqrt(fx2d**2 + fy2d**2)
    return fr

def _hann2d(ny, nx):
    """2D separable Hann apodization."""
    wy = np.hanning(ny)
    wx = np.hanning(nx)
    return np.outer(wy, wx)

def estimate_psd_slope(img, dx=1.0, fmin_frac=0.02, fmax_frac=0.4):
    """
    Estimate beta in P(f) ~ f^-beta from the isotropic 1D PSD (log-log linear fit).
    Uses the provided band [fmin_frac, fmax_frac] * f_max to avoid DC/Nyquist artifacts.
    """
    # 2D periodogram
    F = np.fft.fftshift(np.fft.fft2(img))
    P2D = np.abs(F)**2
    ny, nx = img.shape
    fr = _freq_grids(ny, nx, dx=dx)
    # radial bins
    nbins = min(nx, ny)//2
    f_bins = np.linspace(0.0, fr.max(), nbins+1)
    fr_flat = fr.ravel(); P_flat = P2D.ravel()
    bin_idx = np.digitize(fr_flat, f_bins) - 1
    valid = (bin_idx >= 0) & (bin_idx < nbins)
    sum_bin = np.bincount(bin_idx[valid], weights=P_flat[valid], minlength=nbins)
    cnt_bin = np.bincount(bin_idx[valid], minlength=nbins)
    with np.errstate(invalid="ignore", divide="ignore"):
        Pk = sum_bin / cnt_bin
    Pk[cnt_bin == 0] = np.nan
    f_centers = 0.5*(f_bins[1:] + f_bins[:-1])

    # fit on a safe band
    fmax = np.nanmax(f_centers)
    mask = (f_centers > fmin_frac*fmax) & (f_centers < fmax_frac*fmax) & np.isfinite(Pk) & (Pk > 0)
    x = np.log10(f_centers[mask]); y = np.log10(Pk[mask])
    if x.size < 5:
        return np.nan
    m, b = np.polyfit(x, y, 1)      # log10 P = m log10 f + b
    beta = -m                       # since P ~ f^m -> m = -beta
    return beta

def adjust_psd_slope(img, dx=1.0, delta_beta=0.0, 
                     f_ref=None, band=None, 
                     apodize=True, preserve_mean=True, match_variance=True, eps=None):
    """
    Change the isotropic PSD slope by delta_beta (P -> P * f^{-delta_beta}).
    - delta_beta > 0 : steeper spectrum (more large-scale, smoother image)
    - delta_beta < 0 : flatter/whiter spectrum (more small-scale, rougher image)

    Parameters
    ----------
    img : 2D array
        Input image.
    dx : float
        Pixel size (e.g., meters). Frequencies are cycles per unit of dx.
    delta_beta : float
        Desired slope change: P' ~ P * f^{-delta_beta}.
    f_ref : float or None
        Reference frequency for normalization. If None, use median nonzero f.
    band : tuple (f_lo, f_hi) or None
        If set, apply the slope change only within [f_lo, f_hi] (cycles per unit); smooth edges.
    apodize : bool
        Apply 2D Hann window before FFT to reduce edge ringing.
    preserve_mean : bool
        Keep DC (mean) unchanged.
    match_variance : bool
        Rescale output to match input variance.
    eps : float or None
        Small positive to protect f=0. If None, set to 1/(max(n)*dx).

    Returns
    -------
    out : 2D array (real)
        Image with adjusted spectrum slope.
    """
    img = np.asarray(img, float)
    ny, nx = img.shape

    # Apodization
    if apodize:
        w2 = _hann2d(ny, nx)
        imgw = img * w2
    else:
        imgw = img

    # FFT
    F = np.fft.fftshift(np.fft.fft2(imgw))

    # Radial frequency grid
    fr = _freq_grids(ny, nx, dx=dx)
    if eps is None:
        eps = 1.0 / (max(nx, ny) * dx)

    # Reference frequency
    if f_ref is None:
        f_ref = np.median(fr[fr > 0])

    # Base radial gain for amplitudes (half the PSD exponent)
    H = ((fr + eps) / (f_ref + eps)) ** (-0.5 * delta_beta)

    # Optional band-limiting with smooth cosine tapers
    if band is not None:
        f_lo, f_hi = band
        if f_lo is None: f_lo = 0.0
        if f_hi is None: f_hi = fr.max()
        # smooth 0..1 mask between f_lo and f_hi (raised-cosine of width 10% band)
        width = 0.1 * (f_hi - f_lo) if f_hi > f_lo else 0.0
        def smooth_step(f, a, b):
            # 0 below a, 1 above b, cosine ramp in between
            if b <= a: 
                return (f >= b).astype(float)
            t = np.clip((f - a) / (b - a), 0, 1)
            return 0.5 - 0.5*np.cos(np.pi*t)
        mask_lo = smooth_step(fr, f_lo - width, f_lo + width)
        mask_hi = 1.0 - smooth_step(fr, f_hi - width, f_hi + width)
        band_mask = mask_lo * mask_hi
        H = 1.0 + band_mask * (H - 1.0)

    # Preserve DC (mean) if requested
    if preserve_mean:
        H[fr == 0] = 1.0

    # Apply filter on Fourier amplitudes
    Ff = F * H

    # Inverse FFT (undo shift)
    out = np.fft.ifft2(np.fft.ifftshift(Ff)).real

    # Undo apodization bias (optional): we keep variance matching which is simpler/robust
    if match_variance:
        s_in = np.std(img)
        s_out = np.std(out)
        if s_out > 0:
            out = (out - out.mean()) * (s_in / s_out) + (img.mean() if preserve_mean else 0.0)

    return out

# --- 1) Lat/Lon -> fractional XYZ tile coords at zoom z ---
def latlon_to_xyz_frac(lat, lon, z):
    """Return fractional tile coordinates (xf, yf) at zoom z (Web Mercator)."""
    n = 2 ** z
    xf = (lon + 180.0) / 360.0 * n
    lat_rad = np.radians(lat)
    yf = (1.0 - np.log(np.tan(lat_rad) + 1/np.cos(lat_rad)) / math.pi) / 2.0 * n
    return xf, yf

# --- 2) Fractional tile coords -> (xtile, ytile, px, py) inside 256×256 tile ---
def xyz_frac_to_tile_pixel(xf, yf, tile_size=256):
    xtile = np.floor(xf).astype(int)
    ytile = np.floor(yf).astype(int)
    px = np.floor((xf - xtile) * tile_size).astype(int)
    py = np.floor((yf - ytile) * tile_size).astype(int)
    # clamp just in case of edge cases at tile borders
    px = np.clip(px, 0, tile_size - 1)
    py = np.clip(py, 0, tile_size - 1)
    return xtile, ytile, px, py

# --- 3) Simple tile cache + fetcher for Esri World Imagery ---
ESRI_WORLD_IMAGERY = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

class TileCache:
    def __init__(self):
        self.cache = {}  # (z,x,y) -> PIL.Image
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "research-sampler/1.0"})
    def get_tile(self, z, x, y, timeout=10):
        key = (z, x, y)
        if key in self.cache:
            return self.cache[key]
        url = ESRI_WORLD_IMAGERY.format(z=z, x=x, y=y)
        r = self.session.get(url, timeout=timeout)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        self.cache[key] = img
        return img

# --- 4) Main sampler ---
def sample_esri_world_imagery(lat, lon, zoom=17, tile_size=256):
    """
    lat, lon: arrays of shape (N,)
    zoom: Web Mercator zoom level
    returns: RGB uint8 array of shape (N, 3)
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    assert lat.shape == lon.shape
    N = lat.size

    xf, yf = latlon_to_xyz_frac(lat, lon, zoom)
    xt, yt, px, py = xyz_frac_to_tile_pixel(xf, yf, tile_size=tile_size)

    # Group by tile to fetch each only once
    tile_cache = TileCache()
    rgb = np.zeros((N, 3), dtype=np.uint8)

    # Make an index per unique tile
    tiles, inv = np.unique(np.stack([xt, yt], axis=1), axis=0, return_inverse=True)
    # For each unique tile, fetch and sample all points in that tile
    for t_idx, (x_tile, y_tile) in enumerate(tiles):
        # gather original indices belonging to this tile
        sel = np.where(inv == t_idx)[0]
        # fetch image
        try:
            img = tile_cache.get_tile(zoom, x_tile, y_tile)
        except Exception as e:
            # If a tile fails, leave zeros or handle as you wish
            print(f"Warning: failed to fetch tile z={zoom} x={x_tile} y={y_tile}: {e}")
            continue
        pix = img.load()  # pixel accessor is fast enough for sparse samples
        for i in sel:
            rgb[i] = pix[int(px[i]), int(py[i])]
    return rgb

# ---------- EXAMPLE ----------
# latN, lonN are your arrays of length N
# latN = np.array([...], dtype=float)
# lonN = np.array([...], dtype=float)
# zoom = 17  # adjust to your scale needs
# vals_rgb = sample_esri_world_imagery(latN, lonN, zoom=zoom)
# vals_rgb.shape -> (N, 3)

import numpy as np
import healpy as hp

# --- NESTED helpers ---
def _compact_bits_u64(z):
    z = z & np.uint64(0x5555555555555555)
    z = (z | (z >> 1)) & np.uint64(0x3333333333333333)
    z = (z | (z >> 2)) & np.uint64(0x0F0F0F0F0F0F0F0F)
    z = (z | (z >> 4)) & np.uint64(0x00FF00FF00FF00FF)
    z = (z | (z >> 8)) & np.uint64(0x0000FFFF0000FFFF)
    z = (z | (z >> 16)) & np.uint64(0x00000000FFFFFFFF)
    return z

def _spread_bits_u64(v):
    v = v & np.uint64(0x00000000FFFFFFFF)
    v = (v | (v << 16)) & np.uint64(0x0000FFFF0000FFFF)
    v = (v | (v << 8)) & np.uint64(0x00FF00FF00FF00FF)
    v = (v | (v << 4)) & np.uint64(0x0F0F0F0F0F0F0F0F)
    v = (v | (v << 2)) & np.uint64(0x3333333333333333)
    v = (v | (v << 1)) & np.uint64(0x5555555555555555)
    return v

def _nest_to_fxy(ipix, nside):
    ipix = ipix.astype(np.uint64)
    pp = np.uint64(nside) * np.uint64(nside)
    face = ipix // pp
    inface = ipix % pp
    x = _compact_bits_u64(inface)
    y = _compact_bits_u64(inface >> np.uint64(1))
    return face.astype(np.int64), x.astype(np.int64), y.astype(np.int64)

def _fxy_to_nest(face, x, y, nside):
    inter = _spread_bits_u64(x.astype(np.uint64)) | (_spread_bits_u64(y.astype(np.uint64)) << np.uint64(1))
    base  = face.astype(np.uint64) * (np.uint64(nside) * np.uint64(nside))
    return (base + inter).astype(np.int64)

# --- Main ---
def get_half_interp_weights_ang_general(nside_full, theta, phi, edge_mode="nearest"):
    """
    Bilinear weights from the 'half-level' lattice (EVEN pixels of full grid) to arbitrary directions.
    Returns I (4 even NESTED ids) and W (4 weights summing to 1).
    """
    theta = np.asarray(theta, dtype=np.float64).ravel()
    phi   = np.asarray(phi,   dtype=np.float64).ravel()
    N     = theta.size

    # 1) Use the full-grid containing pixel to locate face/x/y neighborhood
    ids_full = hp.ang2pix(nside_full, theta, phi, nest=True)
    face, x, y = _nest_to_fxy(ids_full, nside_full)

    # 2) Even anchors for 2×2 block on the even lattice
    x0 = (x // 2) * 2
    y0 = (y // 2) * 2
    x1 = x0 + 2
    y1 = y0 + 2

    Xs = [x0, x1, x0, x1]
    Ys = [y0, y0, y1, y1]

    if edge_mode == "nearest":
        Xs = [np.clip(X, 0, nside_full - 1) for X in Xs]
        Ys = [np.clip(Y, 0, nside_full - 1) for Y in Ys]
        drop_mask = [np.zeros(N, dtype=bool) for _ in range(4)]
    elif edge_mode == "drop":
        drop_mask = [
            (Xs[0] < 0) | (Xs[0] >= nside_full) | (Ys[0] < 0) | (Ys[0] >= nside_full),
            (Xs[1] < 0) | (Xs[1] >= nside_full) | (Ys[1] < 0) | (Ys[1] >= nside_full),
            (Xs[2] < 0) | (Xs[2] >= nside_full) | (Ys[2] < 0) | (Ys[2] >= nside_full),
            (Xs[3] < 0) | (Xs[3] >= nside_full) | (Ys[3] < 0) | (Ys[3] >= nside_full),
        ]
        Xs = [np.clip(X, 0, nside_full - 1) for X in Xs]
        Ys = [np.clip(Y, 0, nside_full - 1) for Y in Ys]
    else:
        raise ValueError("edge_mode must be 'nearest' or 'drop'")

    # 3) Map the four even corners to ids
    I = np.empty((4, N), dtype=np.int64)
    for k in range(4):
        I[k] = _fxy_to_nest(face, Xs[k], Ys[k], nside_full)

    # 4) Build 3D vectors (STACK tuples -> (N,3))
    v_tgt = np.vstack(hp.ang2vec(theta, phi,nest=True)).T              # (N,3)
    v00   = np.vstack(hp.pix2vec(nside_full, I[0], nest=True)).T
    v10   = np.vstack(hp.pix2vec(nside_full, I[1], nest=True)).T
    v01   = np.vstack(hp.pix2vec(nside_full, I[2], nest=True)).T
    v11   = np.vstack(hp.pix2vec(nside_full, I[3], nest=True)).T

    # 5) Tangent-plane basis at average corner direction (robust)
    v_c = v00 + v10 + v01 + v11
    v_c /= np.linalg.norm(v_c, axis=1, keepdims=True) + 1e-15

    tmp = v10 - v00
    tmp -= (tmp * v_c).sum(1, keepdims=True) * v_c
    # if degenerate, pick an arbitrary perpendicular
    bad = (np.linalg.norm(tmp, axis=1) < 1e-12)
    if np.any(bad):
        ref = np.zeros_like(v_c)
        ref[:, 0] = 1.0
        # if nearly collinear, use y-axis
        mask = (np.abs((ref * v_c).sum(1)) > 0.99)
        ref[mask] = np.array([0.0, 1.0, 0.0])
        tmp[bad] = ref[bad] - (ref[bad] * v_c[bad]).sum(1, keepdims=True) * v_c[bad]

    e1 = tmp / (np.linalg.norm(tmp, axis=1, keepdims=True) + 1e-15)
    e2 = np.cross(v_c, e1)

    def proj(v):  # (N,3) -> (N,2)
        return np.stack([(v * e1).sum(1), (v * e2).sum(1)], axis=1)

    p_tgt = proj(v_tgt)
    p00   = proj(v00)
    p10   = proj(v10)
    p01   = proj(v01)
    p11   = proj(v11)

    # 6) Solve for (tx, ty) in bilinear map via GN refinement
    a = p10 - p00
    b = p01 - p00
    c = p11 - p10 - p01 + p00
    rhs = p_tgt - p00

    AtA_00 = (a * a).sum(1)
    AtA_11 = (b * b).sum(1)
    AtA_01 = (a * b).sum(1)
    det    = AtA_00 * AtA_11 - AtA_01 * AtA_01
    det[det == 0] = 1e-15
    At_rhs0 = (a * rhs).sum(1)
    At_rhs1 = (b * rhs).sum(1)
    tx = ( AtA_11 * At_rhs0 - AtA_01 * At_rhs1) / det
    ty = (-AtA_01 * At_rhs0 + AtA_00 * At_rhs1) / det

    P_est = p00 + a * tx[:, None] + b * ty[:, None] + c * (tx * ty)[:, None]
    res   = rhs - (P_est - p00)
    J0 = a + c * ty[:, None]
    J1 = b + c * tx[:, None]
    JTJ_00 = (J0 * J0).sum(1)
    JTJ_11 = (J1 * J1).sum(1)
    JTJ_01 = (J0 * J1).sum(1)
    det2   = JTJ_00 * JTJ_11 - JTJ_01 * JTJ_01
    det2[det2 == 0] = 1e-15
    JTres0 = (J0 * res).sum(1)
    JTres1 = (J1 * res).sum(1)
    dtx = ( JTJ_11 * JTres0 - JTJ_01 * JTres1) / det2
    dty = (-JTJ_01 * JTres0 + JTJ_00 * JTres1) / det2
    tx += dtx
    ty += dty

    tx = np.clip(tx, 0.0, 1.0)
    ty = np.clip(ty, 0.0, 1.0)

    # 7) Bilinear weights
    W = np.empty((4, N), dtype=np.float64)
    W[0] = (1 - tx) * (1 - ty)
    W[1] = tx * (1 - ty)
    W[2] = (1 - tx) * ty
    W[3] = tx * ty

    if edge_mode == "drop":
        for k in range(4):
            W[k, drop_mask[k]] = 0.0
        s = W.sum(axis=0)
        ok = s > 0
        W[:, ok] /= s[ok]
        
    return I, W


def conjugate_gradient_normal_equation(data, x0, www, all_idx,
                                       LPT=None,
                                       LP=None,
                                       max_iter=100,
                                       tol=1e-8,
                                       verbose=True):
    """
    Solve the normal equation (Pᵗ P) x = Pᵗ y using the Conjugate Gradient method.
    
    Parameters
    ----------
    data    : array_like
        Observed UTM data y ∈ ℝᵐ
    x0      : array_like
        Initial guess for solution x ∈ ℝⁿ (HEALPix domain)
    www     : interpolation weights
    all_idx : interpolation indices
    LPT     : implementation of adjoint operator Pᵗ
    LP      : implementation of forward operator P
    max_iter: maximum number of CG iterations
    tol     : stopping tolerance on residual norm
    verbose : print convergence info every 50 iterations
    
    Returns
    -------
    x : estimated HEALPix solution u ∈ ℝⁿ
    """

    
    def default_P(x, W, indices):
        """
        Forward operator: P(x) = projection of HEALPix map x onto the UTM grid.
        
        Steps:
        - Apply spherical convolution with kernel w(x,y).
        - Interpolate from HEALPix cells to UTM pixels using weights W and indices.
        """
        return np.sum(x[indices] * W, 0)
    
    def default_PT(y, W, indices, hit):
        """
        Adjoint operator: Pᵗ(y) = back-projection from UTM grid to HEALPix cells.
        
        Steps:
        - Distribute UTM values y back onto contributing HEALPix cells using W.
        - Apply hit normalization (inverse of pixel coverage).
        - Apply spherical convolution with kernel w(x,y).
        """
        value = np.bincount(indices.flatten(),
                            weights=(W * y[None,:]).flatten()) * hit
        return value
            
    if LPT is None:
        LP=default_P
        LPT=default_PT
        
    x = x0.copy()

    # Compute pixel coverage normalization (hit map)
    hit = np.bincount(all_idx.flatten(), weights=www.flatten())
    hit[hit > 0] = 1 / hit[hit > 0]

    # Compute b = Pᵗ y
    b = LPT(data, www, all_idx, hit)

    # Compute initial residual r = b - A x, with A = Pᵗ P
    Ax = LPT(LP(x, www, all_idx), www, all_idx, hit)
    r = b - Ax

    # Initialize direction
    p = r.copy()
    rs_old = np.dot(r, r)

    for i in range(max_iter):
        # Compute A p = Pᵗ P p
        Ap = LPT(LP(p, www, all_idx), www, all_idx, hit)

        alpha = rs_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap

        rs_new = np.dot(r, r)

        if verbose and i % 10 == 0:
            v=np.mean((LP(p, www, all_idx)-data)**2)
            print(f"Iter {i:03d}: residual = {np.sqrt(rs_new):.3e},{np.sqrt(v):.3e}")

        if np.sqrt(rs_new) < tol:
            if verbose:
                print(f"Converged. Iter {i:03d}: residual = {np.sqrt(rs_new):.3e}")
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x


def spectrum_polar_to_cartesian(
    w,
    scales=None,                    # radial *values* (see scale_kind)
    orientations=None,              # angles in radians (uniform)
    n_pixels=512,
    r_max=None,
    method="bilinear",
    fill_value=0.0,
    *,
    scale_kind="frequency",         # "frequency" or "size"
    size_to_freq_factor=1.0,        # if scale_kind="size": freq = size_to_freq_factor / size
):
    """
    If scale_kind == "frequency":
        `scales` are already radii in frequency units, strictly increasing (low->high freq).
    If scale_kind == "size":
        `scales` are spatial sizes (e.g., km, px), strictly increasing (small->large size),
        and they are converted to frequency radii by: freq = size_to_freq_factor / size.
        Choose size_to_freq_factor to get the units you want (e.g., 1.0 for cycles/size).
    """
    from math import pi
    w = np.asarray(w)
    if w.ndim != 2:
        raise ValueError("w must be (Nscale, Norientation)")
    ns, no = w.shape

    # ---- handle scales ----
    if scales is None:
        # default dyadic: sizes OR frequencies depending on scale_kind
        base = 2.0 ** np.arange(ns, dtype=float)
        if scale_kind == "frequency":
            scales = base                     # 1,2,4,... as frequencies
        elif scale_kind == "size":
            # sizes: 1,2,4,...  -> convert to frequency
            scales = size_to_freq_factor / base
        else:
            raise ValueError("scale_kind must be 'frequency' or 'size'")
    else:
        scales = np.asarray(scales, dtype=float)
        if len(scales) != ns:
            raise ValueError("len(scales) must match Nscale")
        if scale_kind == "frequency":
            pass  # already radii
        elif scale_kind == "size":
            # convert sizes -> frequency radii
            scales = size_to_freq_factor / scales
        else:
            raise ValueError("scale_kind must be 'frequency' or 'size'")

    # After conversion, we need strictly increasing radii (low->high frequency)
    if not np.all(np.diff(scales) > 0):
        # If your provided sizes were increasing, 1/size is decreasing => reverse order
        # and reorder w accordingly along the radial axis.
        order = np.argsort(scales)
        scales = scales[order]
        w = w[order, :]

    # ---- orientations (uniform over [0, 2π) ) ----
    if orientations is None:
        orientations = np.linspace(0.0, 2*np.pi, no, endpoint=False)
    else:
        orientations = np.asarray(orientations, dtype=float)
        if len(orientations) != no:
            raise ValueError("len(orientations) must match Norientation")

    # ---- call the previous core (unchanged) ----
    return _spectrum_polar_to_cartesian_core(
        w, scales, orientations, n_pixels, r_max, method, fill_value
    )

def _spectrum_polar_to_cartesian_core(
    w, scales, orientations, n_pixels, r_max, method, fill_value
):
    """Core function from before (unchanged logic), expects increasing frequency radii."""
    ns, no = w.shape
    if r_max is None:
        r_max = float(np.max(scales))

    kx = np.linspace(-r_max, r_max, n_pixels, dtype=float)
    ky = np.linspace(-r_max, r_max, n_pixels, dtype=float)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    R = np.hypot(KX, KY)
    Theta = -np.mod(np.arctan2(KY, KX), 2.0*np.pi)

    radial_index = np.interp(R, scales, np.arange(ns, dtype=float),
                             left=np.nan, right=np.nan)
    dtheta = (2.0*np.pi) / no
    angular_index = Theta / dtheta

    valid = np.isfinite(radial_index)
    try:
        from scipy.ndimage import map_coordinates

        if method.lower() == "bicubic":
            # ===== Bicubic with circular angular wrap =====
            # Pad the angular axis by K columns on both sides so that the cubic kernel
            # has valid neighbors across the 0°/360° seam. K=2 is enough for a cubic kernel.
            K = 2
            # w has shape (Nscale, Norient)
            w_pad = np.concatenate([w[:, -K:], w, w[:, :K]], axis=1)  # (ns, no+2K)

            # Build coordinates for map_coordinates on the padded array:
            # - radial index stays the same, clipped to [0, ns-1] (no wrap)
            # - angular index is shifted by +K and wrapped into [0, no) before querying
            order = 3
            coords = np.vstack([radial_index.ravel(), angular_index.ravel()])

            # clip the radial coordinate to the valid [eps, ns-1-eps] band
            eps = 1e-6
            coords[0, :] = np.where(
                np.isfinite(coords[0, :]),
                np.clip(coords[0, :], 0.0 + eps, (ns - 1) - eps),
                0.0,
            )

            # wrap angular coordinate to [0, no) then shift by +K to address the padded array
            ang = np.mod(coords[1, :], float(no)) + K  # [K, no+K)
            coords[1, :] = ang

            # Now sample the padded array; 'nearest' mode is fine thanks to explicit padding
            sampled = map_coordinates(
                w_pad, coords, order=order, mode="nearest", cval=fill_value, prefilter=True
            ).reshape(n_pixels, n_pixels)

            img = np.where(valid, sampled, fill_value)

        else:
            # ===== Bilinear (or other) without special padding =====
            order = 1
            coords = np.vstack([radial_index.ravel(), angular_index.ravel()])
            eps = 1e-6
            coords[0, :] = np.where(
                np.isfinite(coords[0, :]),
                np.clip(coords[0, :], 0.0 + eps, (ns - 1) - eps),
                0.0,
            )
            # For non-bicubic, SciPy's mode="wrap" is sufficient on the angular axis
            sampled = map_coordinates(
                w, coords, order=order, mode="wrap", cval=fill_value, prefilter=True
            ).reshape(n_pixels, n_pixels)
            img = np.where(valid, sampled, fill_value)

    except Exception:
        # ---- Vectorized bilinear fallback with explicit angular wrap ----
        r_idx = np.floor(radial_index).astype(np.int64)
        t_idx = np.floor(angular_index).astype(np.int64)
        r_idx = np.clip(r_idx, 0, ns - 2)
        t0 = np.mod(t_idx, no)
        t1 = np.mod(t_idx + 1, no)
        tr = np.clip(radial_index - r_idx, 0.0, 1.0)
        ta = np.clip(angular_index - t_idx, 0.0, 1.0)
        f00 = w[r_idx,     t0]
        f01 = w[r_idx,     t1]
        f10 = w[r_idx + 1, t0]
        f11 = w[r_idx + 1, t1]
        g0 = (1.0 - ta) * f00 + ta * f01
        g1 = (1.0 - ta) * f10 + ta * f11
        img = (1.0 - tr) * g0 + tr * g1
        img = np.where(valid, img, fill_value)

    return img, kx, ky

def plot_wave(wave,title="spectrum",unit="Amplitude",cmap="viridis"):
    img, kx, ky = spectrum_polar_to_cartesian(
        wave,
        scales=2**np.arange(wave.shape[0]),                 # tailles croissantes
        scale_kind="size",            # conversion automatique vers fréquence
        size_to_freq_factor=50.0,      # cycles / (unit of size) (Sentinel-2 10m résolution ~to 20m resoltuion for smaller scale; equiv. 50 cycles/km
        method="bicubic",
        n_pixels=512,
    )
    plt.imshow(
            img,
            extent=[kx[0], kx[-1], ky[0], ky[-1]],
            origin="lower",
            aspect="equal",
            cmap=cmap,
    )
    plt.colorbar(label=unit,shrink=0.5)
    plt.xlabel(r"$k_x$ [cycles / km]")
    plt.ylabel(r"$k_y$ [cycles / km]")
    plt.title(title)
    
def lonlat_edges_from_ref(shape, ref_lon, ref_lat, dlon, dlat, anchor="center"):
    """
    Build lon/lat *edges* (H+1, W+1) for a regular, axis-aligned grid.

    Parameters
    ----------
    shape : tuple(int, int)
        (H, W) of the image.
    ref_lon, ref_lat : float
        Reference coordinate in degrees. Interpreted according to `anchor`.
    dlon, dlat : float
        Pixel size in degrees along x (lon) and y (lat). Use positives.
    anchor : {"center","topleft","topright","bottomleft","bottomright"}
        Where (ref_lon, ref_lat) sits relative to the image.

    Returns
    -------
    lon_edges, lat_edges : 2D arrays of shape (H+1, W+1)
        Corner coordinates suitable for `pcolormesh`.
    """
    H, W = shape
    dlon = float(dlon)
    dlat = float(dlat)

    # center of the grid in lon/lat
    if anchor == "center":
        lon0 = ref_lon
        lat0 = ref_lat
    elif anchor == "topleft":
        lon0 = ref_lon + (W/2.0 - 0.5) * dlon
        lat0 = ref_lat - (H/2.0 - 0.5) * dlat
    elif anchor == "topright":
        lon0 = ref_lon - (W/2.0 - 0.5) * dlon
        lat0 = ref_lat - (H/2.0 - 0.5) * dlat
    elif anchor == "bottomleft":
        lon0 = ref_lon + (W/2.0 - 0.5) * dlon
        lat0 = ref_lat + (H/2.0 - 0.5) * dlat
    elif anchor == "bottomright":
        lon0 = ref_lon - (W/2.0 - 0.5) * dlon
        lat0 = ref_lat + (H/2.0 - 0.5) * dlat
    else:
        raise ValueError("anchor must be one of: center/topleft/topright/bottomleft/bottomright")

    # 1D edges (corners) along lon/lat, centered on (lon0, lat0)
    lon_edges_1d = lon0 + (np.arange(W + 1) - W/2.0) * dlon
    lat_edges_1d = lat0 + (np.arange(H + 1) - H/2.0) * dlat

    # 2D corner grids (H+1, W+1)
    lon_edges, lat_edges = np.meshgrid(lon_edges_1d, lat_edges_1d, indexing="xy")
    return lon_edges, lat_edges


def plot_image_lonlat(img, lon_edges, lat_edges, cmap="viridis", vmin=None, vmax=None):
    """
    Plot a 2D image on a lon/lat grid using pcolormesh (no reprojection).
    """
    #fig, ax = plt.subplots(figsize=(7, 5))
    m = plt.pcolormesh(lon_edges, lat_edges, img, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
    #plt.colorbar(m, ax=ax, label="Intensity")
    #ax.set_xlabel("Longitude (deg)")
    #ax.set_ylabel("Latitude (deg)")
    #ax.set_aspect("equal")  # keeps degrees square; remove if you prefer auto
    # add a small margin
    #ax.set_xlim(lon_edges.min(), lon_edges.max())
    #ax.set_ylim(lat_edges.min(), lat_edges.max())
    return fig, ax

import matplotlib.tri as mtri

def _edges_from_centers_2d(C):
    """
    Compute (H+1, W+1) cell-corner coordinates from a 2D array of cell centers (H, W).
    This works for a *structured* grid indexed by (i, j) even if the physical spacing
    is non-uniform and warped.

    Strategy (robust and common in geosciences):
      1) Extrapolate one ghost cell around the array using first-order linear extrapolation.
      2) Corners are the mean of the 2x2 block of surrounding centers in the padded array.

    Parameters
    ----------
    C : (H, W) ndarray
        2D field of *centers* (e.g., lat or lon at pixel centers).

    Returns
    -------
    E : (H+1, W+1) ndarray
        2D field of *edges* (corners), suitable for pcolormesh.
    """
    C = np.asarray(C)
    H, W = C.shape

    # 1) Pad by one cell on all sides using linear extrapolation
    Cp = np.empty((H + 2, W + 2), dtype=C.dtype)
    Cp[1:-1, 1:-1] = C

    # Edges (extrapolate outward along each axis)
    Cp[0, 1:-1]  = 2*C[0, :]  - C[1, :]     # top row
    Cp[-1, 1:-1] = 2*C[-1, :] - C[-2, :]    # bottom row
    Cp[1:-1, 0]  = 2*C[:, 0]  - C[:, 1]     # left col
    Cp[1:-1, -1] = 2*C[:, -1] - C[:, -2]    # right col

    # Corners of the padded array: extrapolate diagonally
    Cp[0, 0]     = 2*C[0, 0]   - C[1, 1]
    Cp[0, -1]    = 2*C[0, -1]  - C[1, -2]
    Cp[-1, 0]    = 2*C[-1, 0]  - C[-2, 1]
    Cp[-1, -1]   = 2*C[-1, -1] - C[-2, -2]

    # 2) Average 2x2 blocks to get corners (H+1, W+1)
    E = 0.25 * (Cp[:-1, :-1] + Cp[1:, :-1] + Cp[:-1, 1:] + Cp[1:, 1:])
    return E


def plot_image_latlon(fig,ax,img, lat, lon, mode="structured", cmap="viridis", vmin=None, vmax=None,
                      shading="flat", aspect="equal"):
    """
    Plot an image given per-pixel lat/lon coordinates.

    Parameters
    ----------
    img : (H, W) ndarray
        Image values per pixel.
    lat, lon : (H, W) ndarray
        Latitude and longitude at *pixel centers* (same shape as `img`).
    mode : {"structured", "scattered"}
        - "structured": (i, j) grid is regular (rectangular index space), possibly warped.
                        We'll compute per-cell corners and use pcolormesh.
        - "scattered" : pixels are not on a regular (i, j) grid. We'll triangulate points and use tripcolor.
    cmap, vmin, vmax : matplotlib colormap settings.
    shading : {"flat","gouraud"} for pcolormesh/tripcolor. "flat" = one color per cell/triangle.
    aspect : matplotlib aspect for axes, e.g. "equal" or "auto".

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    artist  : QuadMesh (structured) or PolyCollection (scattered)
    """
    img = np.asarray(img)
    lat = np.asarray(lat)
    lon = np.asarray(lon)

    if mode == "structured":
        if img.shape != lat.shape or img.shape != lon.shape:
            raise ValueError("For 'structured' mode, img, lat, lon must have the same (H, W) shape.")

        # Compute *corner* grids (H+1, W+1) from center grids (H, W)
        lat_edges = _edges_from_centers_2d(lat)
        lon_edges = _edges_from_centers_2d(lon)

        m = ax.pcolormesh(lon_edges, lat_edges, img, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
        plt.colorbar(m, ax=ax, label="reflectance",shrink=0.5)
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.set_aspect(aspect)
        ax.set_xlim(np.nanmin(lon_edges), np.nanmax(lon_edges))
        ax.set_ylim(np.nanmin(lat_edges), np.nanmax(lat_edges))
        return fig, ax, m

    elif mode == "scattered":
        # Flatten and remove NaNs before triangulation
        z = img.ravel()
        x = lon.ravel()
        y = lat.ravel()
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x, y, z = x[mask], y[mask], z[mask]

        # Triangulate in lon/lat plane
        tri = mtri.Triangulation(x, y)

        fig, ax = plt.subplots(figsize=(7, 5))
        m = ax.tripcolor(tri, z, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
        plt.colorbar(m, ax=ax, label="reflectance",shrink=0.5)
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.set_aspect(aspect)
        ax.set_xlim(np.nanmin(x), np.nanmax(x))
        ax.set_ylim(np.nanmin(y), np.nanmax(y))
        return m

    else:
        raise ValueError("mode must be 'structured' or 'scattered'")
