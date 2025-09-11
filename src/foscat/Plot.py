import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

def lgnomproject(
    cell_ids,               # array-like (N,), HEALPix pixel indices of your samples
    data,                   # array-like (N,), values per cell id
    nside: int,
    rot=None,               # (lon0_deg, lat0_deg, psi_deg). If None: auto-center from cell_ids (pix centers)
    xsize: int = 400,
    ysize: int = 400,
    reso: float = None,     # deg/pixel on tangent plane; if None, use fov_deg
    fov_deg=None,           # full FoV deg (scalar or (fx,fy))
    nest: bool = True,      # True if your cell_ids are NESTED (and ang2pix to be done in NEST)
    reduce: str = "mean",   # 'mean'|'median'|'sum'|'first' when duplicates in cell_ids
    mask_outside: bool = True,
    unseen_value=None,      # default to hp.UNSEEN
    return_image_only: bool = False,
    title: str = None, cmap: str = "viridis", vmin=None, vmax=None,
    notext: bool = False,    # True to avoid tick marks
    hold: bool = True,       # create a new figure if True otherwise use sub to split panel
    sub=(1,1,1),             # declare sub plot
    cbar: bool = False,      # plot colorbar if True

):
    """
    Gnomonic projection from *sparse* HEALPix samples (cell_ids, data) to an image (ysize, xsize).

    For each output image pixel (i,j):
      plane (x,y) --inverse gnomonic--> (theta, phi) --HEALPix--> ipix
      if ipix in `cell_ids`: assign aggregated value, else UNSEEN.

    Parameters
    ----------
    cell_ids : (N,) int array
        HEALPix pixel indices of your samples. Must correspond to `nside` and `nest`.
    data : (N,) float array
        Sample values for each cell id.
    nside : int
        HEALPix NSIDE used for both `cell_ids` and the image reprojection.
    rot : (lon0_deg, lat0_deg, psi_deg) or None
        Gnomonic center (lon, lat) and in-plane rotation psi (deg).
        If None, we auto-center from the *centers of the provided pixels* (via hp.pix2ang).
    xsize, ysize : int
        Output image size (pixels).
    reso : float or None
        Pixel size (deg/pixel) on the tangent plane. If None, derived from `fov_deg`.
    fov_deg : float or (float,float)
        Full field of view in degrees.
    nest : bool
        Use True if your `cell_ids` correspond to NESTED indexing.
    reduce : str
        How to combine duplicate cell ids: 'mean'|'median'|'sum'|'first'.
    mask_outside : bool
        Mask pixels outside the valid gnomonic hemisphere (cosc <= 0).
    unseen_value : float or None
        Value for invalid pixels (default hp.UNSEEN).
    
    return_image_only : bool
        If True, return the 2D array only (no plotting).

    Returns
    -------
    (fig, ax, img) or img
        If return_image_only=True, returns img (ysize, xsize).
    """
    if unseen_value is None:
        unseen_value = hp.UNSEEN

    cell_ids = np.asarray(cell_ids, dtype=np.int64)
    vals_in  = np.asarray(data, dtype=float)
    if cell_ids.shape != vals_in.shape:
        raise ValueError("cell_ids and data must have the same shape (N,)")

    # -------- 1) Aggregate duplicates in cell_ids (if any) --------
    uniq, inv = np.unique(cell_ids, return_inverse=True)  # uniq is sorted
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
        # Center from pixel centers of provided ids
        theta_c, phi_c = hp.pix2ang(nside, uniq, nest=nest)  # colat, lon (rad)
        # circular mean for lon, median for colat
        lon0_deg = np.degrees(np.angle(np.mean(np.exp(1j * phi_c))))
        lat0_deg = 90.0 - np.degrees(np.median(theta_c))
        psi_deg  = 0.0
        rot = (lon0_deg % 360.0, float(lat0_deg), float(psi_deg))

    lon0_deg, lat0_deg, psi_deg = rot
    lon0 = np.deg2rad(lon0_deg)
    lat0 = np.deg2rad(lat0_deg)
    psi  = np.deg2rad(psi_deg)

    # -------- 3) Tangent-plane grid (gnomonic) --------
    if reso is not None:
        dx = np.tan(np.deg2rad(reso))
        dy = dx
        half_x = 0.5 * xsize * dx
        half_y = 0.5 * ysize * dy
    else:
        if fov_deg is None:
            fov_deg=np.rad2deg(np.sqrt(cell_ids.shape[0])/nside)*1.4
            
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
    
    X, Y = np.meshgrid(xs, ys)  # (ysize, xsize)

    # in-plane rotation psi
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

    # -------- 5) Map image pixels to HEALPix ids --------
    ip_img = hp.ang2pix(nside, theta_img.ravel(), lon.ravel(), nest=nest).astype(np.int64)

    # -------- 6) Assign values by matching ip_img ∈ uniq (safe searchsorted) --------
    # uniq is sorted; build insertion pos then check matches only where pos < len(uniq)
    pos = np.searchsorted(uniq, ip_img, side="left")
    valid = pos < uniq.size
    match = np.zeros_like(valid, dtype=bool)
    match[valid] = (uniq[pos[valid]] == ip_img[valid])

    img_flat = np.full(ip_img.shape, unseen_value, dtype=float)
    img_flat[match] = agg_vals[pos[match]]
    img = img_flat.reshape(ysize, xsize)

    # Mask out-of-hemisphere gnomonic region
    if mask_outside:
        img[outside] = unseen_value

    # -------- 7) Return / plot --------
    if return_image_only:
        return img

    # Axes in approx. "gnomonic degrees" (atan of plane coords)
    x_deg = np.degrees(np.arctan(xs))
    y_deg = np.degrees(np.arctan(ys))

    longitude_min=x_deg[0]/np.cos(np.deg2rad(lat0_deg))+lon0_deg
    longitude_max=x_deg[-1]/np.cos(np.deg2rad(lat0_deg))+lon0_deg
    
    if longitude_min>180:
        longitude_min-=360
        longitude_max-=360
    
    extent = (longitude_min,longitude_max,
              y_deg[0]+lat0_deg, y_deg[-1]+lat0_deg)

    if hold:
        fig, ax = plt.subplots(figsize=(xsize/100, ysize/100), dpi=100)
    else:
        ax=plt.subplot(sub[0],sub[1],sub[2])
    
    im = ax.imshow(
        np.where(img == unseen_value, np.nan, img),
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        interpolation="nearest",
        aspect="auto"
    )
    if not notext:
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        
    if title:
        ax.set_title(title)
        
    if cbar:
        if hold:
            cb = fig.colorbar(im, ax=ax)
            cb.set_label("value")
        else:
            plt.colorbar(im, ax=ax, orientation="horizontal", label="value")
        
    plt.tight_layout()
    if hold:
        return fig, ax #, img
    else:
        return ax


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
