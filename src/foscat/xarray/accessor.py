import xarray as xr

from foscat.xarray.statistics import _xarray_to_scat_cov


@xr.register_dataset_accessor("foscat")
class FoscatAccessor:
    def __init__(self, ds):
        self._ds = ds

    def plot(self, name=None, hold=True, color="blue", lw=1, legend=True):
        stats = _xarray_to_scat_cov(self._ds)
        if isinstance(stats, tuple):
            # don't use the variances for now
            stats = stats[0]

        return stats.plot(name=name, hold=hold, color=color, lw=lw, legend=legend)
