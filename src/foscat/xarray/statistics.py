import xarray as xr

import foscat.scat_cov as sc

backends = {1: "tensorflow", 2: "torch", 3: "numpy"}


def _scat_cov_to_xarray(obj, batch_dim="batches"):
    types = xr.Variable("type", ["mean", "variance"])
    names = ["S0", "P00", "C01", "C11", "S1", "C10"]
    dims = {
        "S0": [batch_dim, "type"],
        "S1": [batch_dim, "masks", "scales1", "orientations_1"],
        "P00": [batch_dim, "masks", "scales1", "orientations_1"],
        "C01": [batch_dim, "masks", "scales2", "orientations_1", "orientations_2"],
        "C10": [batch_dim, "masks", "scales2", "orientations_1", "orientations_2"],
        "C11": [
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
        attrs={"foscat_backend": backends[obj.backend.BACKEND], "use_1d": obj.use_1D},
    )


def _xarray_to_scat_cov(ds):
    # TODO: use groupby for this instead?
    stats_ = {
        key: var.variable
        for key, var in ds.data_vars.items()
        if not key.startswith("var_")
    }
    vars_ = {
        key: var.variable for key, var in ds.data_vars.items() if key.startswith("var_")
    }

    kwarg_names = {
        "use_1d": "use_1D",
        "foscat_backend": "backend",
    }
    kwargs = {kwarg_names.get(key, key): value for key, value in ds.attrs.items()}

    stats_kwargs = {key.lower(): var.data for key, var in stats_.items()}
    stats = sc.scat_cov(**stats_kwargs, **kwargs)

    var_kwargs = {key.lower(): var.data for key, var in vars_.items()}
    if var_kwargs:
        variances = sc.scat_cov(**var_kwargs, **kwargs)
        return stats, variances
    else:
        return stats


def reference_statistics(
    arr, parameters, variances=False, mask=None, norm=None, cmat=None
):
    # what does `cmat` stand for? Correlation matrix? And is `auto` important in the autocorrelation (this function)?
    kwargs = {
        "calc_var": variances,
        "mask": mask,
        "norm": norm,
        "cmat": cmat,
    }

    other_dims = [dim for dim in arr.dims if dim != "cells"]
    if not other_dims:
        arr_ = arr
        batch_dim = "batches"
    elif len(other_dims) == 1:
        arr_ = arr
        (batch_dim,) = other_dims
    else:
        batch_dim = "stacked"
        arr_ = arr.stack({batch_dim: other_dims})

    data = arr_.transpose(..., "cells").data
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

    stats = _scat_cov_to_xarray(ref, batch_dim=batch_dim).assign_coords(coords)

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
