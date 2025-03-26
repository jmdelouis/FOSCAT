import xarray as xr

import foscat.scat_cov as sc


def _scat_cov_to_xarray(obj, batch_dim="batches"):
    types = xr.Variable("type", ["mean", "variance"])
    names = ["S0", "S2", "S3", "S4", "S1", "S3P"]
    dims = {
        "S0": [batch_dim, "type"],
        "S1": [batch_dim, "masks", "scales1", "orientations_1"],
        "S2": [batch_dim, "masks", "scales1", "orientations_1"],
        "S3": [batch_dim, "masks", "scales2", "orientations_1", "orientations_2"],
        "S3P": [batch_dim, "masks", "scales2", "orientations_1", "orientations_2"],
        "S4": [
            batch_dim,
            "masks",
            "scales3",
            "orientations_1",
            "orientations_2",
            "orientations_3",
        ],
    }
    data = {name: getattr(obj, name) for name in names}

    return xr.Dataset(
        data_vars={
            name: (dims[name], values)
            for name, values in data.items()
            if values is not None
        },
        coords={"types": types},
        attrs={"foscat_backend": obj.backend.BACKEND, "use_1d": obj.use_1D},
    )


def _xarray_to_scat_cov(ds):
    # TODO: use groupby for this instead?
    stats_ = {
        key: var.variable
        for key, var in ds.data_vars.items()
        if not key.startswith("var_")
    }
    vars_ = {
        key.lower().removeprefix("var_"): var.variable
        for key, var in ds.data_vars.items()
        if key.startswith("var_")
    }

    kwarg_names = {
        "use_1d": "use_1D",
        "foscat_backend": "backend",
    }
    kwargs = {kwarg_names.get(key, key): value for key, value in ds.attrs.items()}

    stats_kwargs = {key.lower(): var.data for key, var in stats_.items()}
    stats = sc.scat_cov(**stats_kwargs, **kwargs)

    var_kwargs = {key: var.data for key, var in vars_.items()}
    if var_kwargs:
        variances = sc.scat_cov(**var_kwargs, **kwargs)
        return stats, variances
    else:
        return stats


def stack_other_dims(arr, spatial_dim, batch_dim):
    other_dims = [dim for dim in arr.dims if dim != spatial_dim]
    if not other_dims:
        arr_ = arr
    elif len(other_dims) == 1:
        arr_ = arr
        (batch_dim,) = other_dims
    else:
        arr_ = arr.stack({batch_dim: other_dims})

    return arr_, other_dims, batch_dim


def reference_statistics(
    arr,
    *,
    parameters,
    spatial_dim="cells",
    variances=False,
    mask=None,
    norm=None,
):
    """
    reference statistics for a single image

    Parameters
    ----------
    arr : xarray.DataArray
        Input image. For now, only 1D healpix is supported. Every dimension other than
        the spatial dimension (see ``spatial_dim``) will be stacked.
    parameters : Parameters
        The parameters for the scattering covariance transform.
    spatial_dim : str, default: "cells"
        The spatial dimension.
    variances : bool, default: False
        Whether to compute the variances of the statistic values.
    mask : xarray.DataArray, optional
        Mask out certain regions. Not implemented yet.
    norm : {"auto", "self"} or None, default: None
        Normalization method:
        - None: no normalization
        - "auto": normalize by the reference S2
        - "self": normalize by the current S2
    """
    if spatial_dim not in arr.dims:
        raise ValueError(
            f"cannot find the spatial dim '{spatial_dim}' in the data dimensions"
        )

    # what does `cmat` stand for? Correlation matrix? And is `auto` important in the autocorrelation (this function)?
    kwargs = {
        "calc_var": variances,
        # "mask": mask,
        "norm": norm,
    }

    arr_, other_dims, batch_dim = stack_other_dims(arr, spatial_dim, "batches")

    data = arr_.transpose(..., spatial_dim).data
    result = parameters.cache.eval(data, **kwargs)
    if result is None:
        raise ValueError(
            "something failed, check the logs"
        )  # TODO: change the foscat code to raise errors

    if variances:
        ref, sref = result
    else:
        ref, sref = result, None

    coords = {
        name: coord
        for name, coord in arr_.coords.items()
        if set(coord.dims).intersection(other_dims)
    }

    stats = (
        _scat_cov_to_xarray(ref, batch_dim=batch_dim)
        .assign_coords(coords)
        .assign_attrs({"other_dims": other_dims})
    )

    if sref is not None:
        variances = _scat_cov_to_xarray(sref, batch_dim=batch_dim)
        stats = stats.merge(
            variances.rename_vars(
                {name: f"var_{name}" for name in variances.data_vars.keys()}
            )
        )

    if not other_dims:
        return stats.squeeze(batch_dim)
    elif len(other_dims) == 1:
        return stats
    else:
        return stats.unstack(batch_dim)


def cross_statistics(
    arr1,
    arr2,
    *,
    parameters,
    spatial_dim="cells",
    variances=False,
    mask=None,
    norm=None,
):
    """
    cross statistics between two images

    Parameters
    ----------
    arr1, arr2 : xarray.DataArray
        Input images. Must align exactly. For now, only 1D healpix is supported. Every
        dimension other than the spatial dimension (see ``spatial_dim``) will be stacked.
    parameters : Parameters
        The parameters for the scattering covariance transform.
    spatial_dim : str, default: "cells"
        The spatial dimension.
    variances : bool, default: False
        Whether to compute the variances of the statistic values.
    mask : xarray.DataArray, optional
        Mask out certain regions. Not implemented yet.
    norm : {"auto", "self"} or None, default: None
        Normalization method:
        - None: no normalization
        - "auto": normalize by the reference S2
        - "self": normalize by the current S2
    """
    # make sure the indexes align exactly (i.e. the arrays only differ by their values)
    xr.align(arr1, arr2, join="exact", copy=False)
    if spatial_dim not in arr1.dims:
        raise ValueError(
            f"cannot find the spatial dim '{spatial_dim}' in the data dimensions"
        )

    kwargs = {
        "calc_var": variances,
        # "mask": mask,
        "norm": norm,
    }

    # will always stack equally
    arr1_, other_dims, batch_dim = stack_other_dims(
        arr1, spatial_dim=spatial_dim, batch_dim="batches"
    )
    arr2_, _, _ = stack_other_dims(arr2, spatial_dim=spatial_dim, batch_dim="batches")

    data1 = arr1_.transpose(..., spatial_dim).data
    data2 = arr1_.transpose(..., spatial_dim).data

    result = parameters.cache.eval(data1, data2, **kwargs)
    if result is None:
        raise ValueError(
            "something failed, check the logs"
        )  # TODO: change the foscat code to raise errors

    if variances:
        ref, sref = result
    else:
        ref, sref = result, None

    coords1 = {
        name: coord
        for name, coord in arr1_.coords.items()
        if set(coord.dims).intersection(other_dims)
    }
    coords2 = {
        name: coord
        for name, coord in arr2_.coords.items()
        if set(coord.dims).intersection(other_dims)
    }
    coords = coords1 | coords2

    stats = (
        _scat_cov_to_xarray(ref, batch_dim=batch_dim)
        .assign_coords(coords)
        .assign_attrs({"other_dims": other_dims})
    )

    if sref is not None:
        variances = _scat_cov_to_xarray(sref, batch_dim=batch_dim)
        stats = stats.merge(
            variances.rename_vars(
                {name: f"var_{name}" for name in variances.data_vars.keys()}
            )
        )

    if not other_dims:
        return stats.squeeze(batch_dim)
    elif len(other_dims) == 1:
        return stats
    else:
        return stats.unstack(batch_dim)
