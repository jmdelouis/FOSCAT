import xarray as xr

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
        attrs={"foscat_backend": backends[obj.backend.BACKEND]},
    )


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
