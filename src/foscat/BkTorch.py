import sys

import numpy as np
import torch
import torch.nn.functional as F

import foscat.BkBase as BackendBase


class BkTorch(BackendBase.BackendBase):

    def __init__(self, *args, **kwargs):
        # Impose que use_2D=True pour la classe scat
        super().__init__(name="torch", *args, **kwargs)
        self.backend = torch
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.float64 = self.backend.float64
        self.float32 = self.backend.float32
        self.int64 = self.backend.int64
        self.int32 = self.backend.int32
        self.complex64 = self.backend.complex128
        self.complex128 = self.backend.complex64

        dtype_map = {
            "float32": (self.backend.float32, self.backend.complex64),
            "float64": (self.backend.float64, self.backend.complex128),
        }

        if self.all_type in dtype_map:
            self.all_bk_type, self.all_cbk_type = dtype_map[self.all_type]
        else:
            raise ValueError(
                f"ERROR INIT foscat: {self.all_type} should be float32 or float64"
            )

        # ===========================================================================
        # INIT
        if self.mpi_rank == 0:
            sys.stdout.flush()

        gpus = torch.cuda.is_available()

        gpuname = "CPU:0"
        self.gpulist = {}
        self.gpulist[0] = gpuname
        self.ngpu = 1

        if gpus:
            try:
                self.ngpu = torch.cuda.device_count()
                self.gpulist = {}
                for k in range(self.ngpu):
                    self.gpulist[k] = torch.cuda.get_device_name(k)

            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        self.torch_device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def downsample_mean_2x2(self, tim: torch.Tensor) -> torch.Tensor:
        """
        Average-pool tensor tim over non-overlapping 2x2 spatial blocks.

        Parameters
        ----------
        tim : torch.Tensor
            Tensor of shape [a, N1, N2, b].

        Returns
        -------
        torch.Tensor
            Downsampled tensor of shape [a, N1//2, N2//2, b],
            each element being the mean of a 2x2 block.
        """
        a, N1, N2, b = tim.shape
        # Ensure even sizes
        N1_2 = N1 // 2
        N2_2 = N2 // 2

        # reshape to group 2x2 patches
        tim_reshaped = tim[:, : 2 * N1_2, : 2 * N2_2, :].reshape(a, N1_2, 2, N2_2, 2, b)
        # mean over the two small dims (2x2)
        out = tim_reshaped.mean(dim=(2, 4))
        return out

    def downsample_median_2x2(self, tim: torch.Tensor) -> torch.Tensor:
        """
        2x2 block median downsampling on spatial axes (N1, N2).

        Input:
            tim: [a, N1, N2, b]  (real or complex)
        Output:
            out: [a, N1//2, N2//2, b]
              each value is the median over the corresponding 2x2 block.
            - For complex inputs: median is taken by sorting the 4 values by |.|,
              returning the complex sample at the lower median rank.
        """
        a, N1, N2, b = tim.shape
        N1_2 = N1 // 2
        N2_2 = N2 // 2
        # On ignore la dernière ligne/colonne si N1/N2 sont impairs
        x = tim[:, : 2 * N1_2, : 2 * N2_2, :]  # [a, 2*N1_2, 2*N2_2, b]

        # Regrouper les blocs 2x2 -> construire une dernière dimension de taille 4
        # Réarrange: [a, N1_2, 2, N2_2, 2, b] -> [a, N1_2, N2_2, b, 4]
        x = (
            x.reshape(a, N1_2, 2, N2_2, 2, b)
            .permute(0, 1, 3, 5, 2, 4)
            .reshape(a, N1_2, N2_2, b, 4)
        )

        if not torch.is_complex(x):
            # Réel : médiane le long de la dernière dim (taille 4)
            med, _ = torch.median(x, dim=-1)  # [a, N1_2, N2_2, b]
            return med
        else:
            # Complexe : trier par module puis prendre l'élément de rang 1 (médiane inférieure)
            mags = x.abs()  # [a, N1_2, N2_2, b, 4]
            sorted_mag, idx = torch.sort(
                mags, dim=-1
            )  # idx: indices triés par |.| croissant
            # Récupérer l'indice de médiane inférieure (pour 4 éléments -> position 1)
            med_rank = 1
            gather_idx = idx[..., med_rank : med_rank + 1]  # [a, N1_2, N2_2, b, 1]
            # Sélectionner la valeur complexe correspondante
            med = torch.gather(x, dim=-1, index=gather_idx).squeeze(
                -1
            )  # [a, N1_2, N2_2, b]
            return med

    def downsample_mean_1d(self, tim: torch.Tensor) -> torch.Tensor:
        """
        Downsample tensor tim [a, N1] by averaging non-overlapping 2-element blocks.
        Output shape: [a, N1//2]
        """
        a, N1 = tim.shape
        N1_2 = N1 // 2

        # Ignore the last element if N1 is odd
        x = tim[:, : 2 * N1_2]

        # Reshape to group pairs of 2 and take mean
        x = x.reshape(a, N1_2, 2)
        out = x.mean(dim=-1)  # [a, N1//2]
        return out

    def downsample_median_1d(self, tim: torch.Tensor) -> torch.Tensor:
        """
        Downsample tensor tim [a, N1] by taking the median of non-overlapping pairs (2 values).
        Output shape: [a, N1//2]
          - For real inputs: median of the two values.
          - For complex inputs: pick the complex value with the smallest |.| among the two.
        """
        a, N1 = tim.shape
        N1_2 = N1 // 2
        x = tim[:, : 2 * N1_2].reshape(a, N1_2, 2)  # group 2 by 2

        if not torch.is_complex(x):
            # Sort values in ascending order, then take mean of the two (true median for 2 samples)
            x_sorted, _ = torch.sort(x, dim=-1)
            med = x_sorted.mean(dim=-1)  # [a, N1//2]
            return med
        else:
            # Complex: sort by magnitude
            mags = x.abs()
            sorted_mags, idx = torch.sort(mags, dim=-1)
            # Take the one with smallest magnitude (lower median)
            med = torch.gather(x, dim=-1, index=idx[..., 0:1]).squeeze(-1)
            return med

    # ---------------------------------
    # HEALPix binning utilities (nested)
    # ---------------------------------
    # Robust binned_mean that supports arbitrary subsets (N not divisible by 4)
    # and batched cell_ids of shape [B, N]. It returns compact per-parent means
    # even when some parents are missing (sparse coverage).

    def binned_mean_old(
        self,
        data,
        cell_ids,
        *,
        reduce: str = "mean",  # <-- NEW: "mean" (par défaut) ou "max"
        padded: bool = False,
        fill_value: float = float("nan"),
    ):
        """
            Reduce values over parent HEALPix pixels (nested) when downgrading nside→nside/2.

        Parameters
            ----------
            data : torch.Tensor | np.ndarray
            Shape [..., N] or [B, ..., N].
            cell_ids : torch.LongTensor | np.ndarray
            Shape [N] or [B, N] (nested indexing at the child resolution).
            reduce : {"mean","max"}, default "mean"
            Aggregation to apply within each parent group of 4 children.
            padded : bool, default False
            Only used when `cell_ids` is [B, N]. If False, returns ragged Python lists.
            If True, returns padded tensors + mask.
            fill_value : float, default NaN
            Padding value when `padded=True`.

        Returns
            -------
            # idem à ta doc existante, mais la valeur est une moyenne (reduce="mean")
            # ou un maximum (reduce="max").
        """

        # ---- Tensorize & device/dtype plumbing ----
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(
                dtype=torch.float32, device=getattr(self, "torch_device", "cpu")
            )
        if isinstance(cell_ids, np.ndarray):
            cell_ids = torch.from_numpy(cell_ids).to(
                dtype=torch.long, device=data.device
            )
            data = data.to(device=getattr(self, "torch_device", data.device))
            cell_ids = cell_ids.to(device=data.device, dtype=torch.long)

        if data.ndim < 1:
            raise ValueError("`data` must have at least 1 dimension (last is N).")
        N = data.shape[-1]

        # Utilitaires pour 'max' (fallback si scatter_reduce_ indisponible)
        def _segment_max(vals_flat, idx_flat, out_size):
            """Retourne out[out_idx] = max(vals[ idx==out_idx ]), vectorisé si possible."""
            # PyTorch >= 1.12 / 2.0: scatter_reduce_ disponible
            if hasattr(torch.Tensor, "scatter_reduce_"):
                out = torch.full(
                    (out_size,),
                    -float("inf"),
                    dtype=vals_flat.dtype,
                    device=vals_flat.device,
                )
                out.scatter_reduce_(
                    0, idx_flat, vals_flat, reduce="amax", include_self=True
                )
                return out
            # Fallback simple (boucle sur indices uniques) – OK pour du downsample
            out = torch.full(
                (out_size,),
                -float("inf"),
                dtype=vals_flat.dtype,
                device=vals_flat.device,
            )
            uniq = torch.unique(idx_flat)
            for u in uniq.tolist():
                m = idx_flat == u
                # éviter max() sur tensor vide
                if m.any():
                    out[u] = torch.max(vals_flat[m])
            return out

        # ---- Flatten leading dims for scatter convenience ----
        orig = data.shape[:-1]
        if cell_ids.ndim == 1:
            # Shared mapping for all rows
            groups = (cell_ids // 4).to(torch.long)  # [N]
            parents, inv = torch.unique(groups, sorted=True, return_inverse=True)
            n_bins = parents.numel()

            R = int(np.prod(orig)) if len(orig) > 0 else 1
            data_flat = data.reshape(R, N)  # [R, N]
            row_offsets = torch.arange(R, device=data.device).unsqueeze(1) * n_bins
            idx = inv.unsqueeze(0).expand(R, -1) + row_offsets  # [R, N]

            vals_flat = data_flat.reshape(-1)
            idx_flat = idx.reshape(-1)
            out_size = R * n_bins

            if reduce == "mean":
                out_sum = torch.zeros(out_size, dtype=data.dtype, device=data.device)
                out_cnt = torch.zeros_like(out_sum)
                out_sum.scatter_add_(0, idx_flat, vals_flat)
                out_cnt.scatter_add_(0, idx_flat, torch.ones_like(vals_flat))
                out_cnt.clamp_(min=1)
                reduced = out_sum / out_cnt
            elif reduce == "max":
                reduced = _segment_max(vals_flat, idx_flat, out_size)
            else:
                raise ValueError("reduce must be 'mean' or 'max'.")

            output = reduced.view(*orig, n_bins)
            return output, parents

        elif cell_ids.ndim == 2:
            # Per-batch mapping
            B = cell_ids.shape[0]
            R = int(np.prod(orig)) if len(orig) > 0 else 1
            data_flat = data.reshape(R, N)  # [R, N]
            B_data = data.shape[0] if len(orig) > 0 else 1
            if B_data % B != 0:
                raise ValueError(
                    f"Leading dim of data ({B_data}) must be a multiple of cell_ids batch ({B})."
                )
            # T = repeats per batch row (product of extra leading dims)
            T = (R // B_data) if B_data > 0 else 1

            means_list, groups_list = [], []
            max_bins = 0

            for b in range(B):
                groups_b = (cell_ids[b] // 4).to(torch.long)  # [N]
                parents_b, inv_b = torch.unique(
                    groups_b, sorted=True, return_inverse=True
                )
                n_bins_b = parents_b.numel()
                max_bins = max(max_bins, n_bins_b)

                # rows for this batch in data_flat
                start, stop = b * T, (b + 1) * T
                rows = slice(start, stop)  # T rows

                row_offsets = (
                    torch.arange(T, device=data.device).unsqueeze(1) * n_bins_b
                )
                idx = inv_b.unsqueeze(0).expand(T, -1) + row_offsets  # [T, N]

                vals_flat = data_flat[rows].reshape(-1)
                idx_flat = idx.reshape(-1)
                out_size = T * n_bins_b

                if reduce == "mean":
                    out_sum = torch.zeros(
                        out_size, dtype=data.dtype, device=data.device
                    )
                    out_cnt = torch.zeros_like(out_sum)
                    out_sum.scatter_add_(0, idx_flat, vals_flat)
                    out_cnt.scatter_add_(0, idx_flat, torch.ones_like(vals_flat))
                    out_cnt.clamp_(min=1)
                    reduced_bt = (out_sum / out_cnt).view(T, n_bins_b)
                elif reduce == "max":
                    reduced_bt = _segment_max(vals_flat, idx_flat, out_size).view(
                        T, n_bins_b
                    )
                else:
                    raise ValueError("reduce must be 'mean' or 'max'.")

                means_list.append(reduced_bt)
                groups_list.append(parents_b)

            if not padded:
                return means_list, groups_list

            # Padded output (B, T, max_bins) [+ mask]
            mean_pad = torch.full(
                (B, T, max_bins), fill_value, dtype=data.dtype, device=data.device
            )
            groups_pad = torch.full(
                (B, max_bins), -1, dtype=torch.long, device=data.device
            )
            mask = torch.zeros((B, max_bins), dtype=torch.bool, device=data.device)
            for b, (m_b, g_b) in enumerate(zip(means_list, groups_list)):
                nb = g_b.numel()
                mean_pad[b, :, :nb] = m_b
                groups_pad[b, :nb] = g_b
                mask[b, :nb] = True

            # Reshape back to [B, (*extra dims), max_bins] si besoin
            if len(orig) > 1:
                extra = orig[1:]
                mean_pad = mean_pad.view(B, *extra, max_bins)
            else:
                mean_pad = mean_pad.view(B, max_bins)

            return mean_pad, groups_pad, mask

        else:
            raise ValueError("`cell_ids` must be of shape [N] or [B, N].")

    def binned_mean(  # (garde ton nom si besoin de compat)
        self,
        data,
        cell_ids,
        *,
        reduce: str = "mean",  # "mean" | "max" | "median"
        padded: bool = False,
        fill_value: float = float("nan"),
    ):
        """
        Reduce values over parent HEALPix pixels (nested) when downgrading nside→nside/2.

        Parameters
        ----------
        data : torch.Tensor | np.ndarray
            Shape [..., N] or [B, ..., N].
        cell_ids : torch.LongTensor | np.ndarray
            Shape [N] or [B, N] (nested indexing at the child resolution).
        reduce : {"mean","max","median"}, default "mean"
            Aggregation within each parent group of 4 children.
        padded : bool, default False
            Only when `cell_ids` is [B, N]. If False, returns ragged Python lists.
            If True, returns padded tensors + mask.
        fill_value : float, default NaN
            Padding value when `padded=True`.

        Returns
        -------
        As in your original function, with aggregation set by `reduce`.
        """

        # ---- Tensorize & device/dtype plumbing ----
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(
                dtype=torch.float32, device=getattr(self, "torch_device", "cpu")
            )
        if isinstance(cell_ids, np.ndarray):
            cell_ids = torch.from_numpy(cell_ids).to(
                dtype=torch.long, device=getattr(self, "torch_device", data.device)
            )
        data = data.to(device=getattr(self, "torch_device", data.device))
        cell_ids = cell_ids.to(device=data.device, dtype=torch.long)

        if data.ndim < 1:
            raise ValueError("`data` must have at least 1 dimension (last is N).")
        N = data.shape[-1]
        orig = data.shape[:-1]

        # ---- Utilities ----
        def _segment_max(vals_flat, idx_flat, out_size):
            """Compute out[g] = max(vals[idx==g]); vectorized when possible."""
            if hasattr(torch.Tensor, "scatter_reduce_"):
                out = torch.full(
                    (out_size,),
                    -float("inf"),
                    dtype=vals_flat.dtype,
                    device=vals_flat.device,
                )
                out.scatter_reduce_(
                    0, idx_flat, vals_flat, reduce="amax", include_self=True
                )
                return out
            out = torch.full(
                (out_size,),
                -float("inf"),
                dtype=vals_flat.dtype,
                device=vals_flat.device,
            )
            uniq = torch.unique(idx_flat)
            for u in uniq.tolist():
                m = idx_flat == u
                if m.any():
                    out[u] = torch.max(vals_flat[m])
            return out

        def _median_from_four_real(v4: torch.Tensor) -> torch.Tensor:
            """
            v4: [..., 4] real with NaN for missing.
            Returns true median: average of the 2 middle finite values.
            """
            # Sort with NaN-last: replace NaN by +inf for sorting, then restore
            is_nan = torch.isnan(v4)
            v4_sortkey = torch.where(is_nan, torch.full_like(v4, float("inf")), v4)
            v_sorted, _ = torch.sort(v4_sortkey, dim=-1)  # NaN (inf) at the end

            # Count finite per group
            k = torch.sum(~is_nan, dim=-1)  # [...]

            # For k==0 -> NaN; k==1 -> the single value; k>=2 -> average middle two
            # Indices for middle-two among the first k finite values: m-1 and m (with m = k//2)
            m = torch.clamp(k // 2, min=1)  # ensure >=1 for gather
            idx_lo = (m - 1).unsqueeze(-1)
            idx_hi = torch.clamp(m, max=3).unsqueeze(-1)  # upper middle (cap at 3)

            # Gather from sorted finite section
            gather_lo = torch.gather(v_sorted, -1, idx_lo).squeeze(-1)
            gather_hi = torch.gather(v_sorted, -1, idx_hi).squeeze(-1)
            med = 0.5 * (gather_lo + gather_hi)

            # Handle k==1: both idx point to same single finite value -> OK
            # Handle k==0: set NaN
            med = torch.where(k > 0, med, torch.full_like(med, float("nan")))
            return med

        def _median_from_four_complex(v4: torch.Tensor) -> torch.Tensor:
            """
            v4: [..., 4] complex with NaN for missing.
            Returns lower median by magnitude (rank 1 among finite elements).
            """
            mags = v4.abs()
            # NaN mags -> set to +inf so they go last
            mags_key = torch.where(
                torch.isnan(mags), torch.full_like(mags, float("inf")), mags
            )
            mags_sorted, idx = torch.sort(mags_key, dim=-1)
            # Count finite elements
            k = torch.sum(torch.isfinite(mags), dim=-1)  # [...]
            # lower median rank = max(0, k//2 - 1)
            rank = torch.clamp(k // 2 - 1, min=0).unsqueeze(-1)
            pick = torch.gather(idx, -1, rank)
            med = torch.gather(v4, -1, pick).squeeze(-1)
            # If k==0 -> NaN+NaNj
            med = torch.where(
                k > 0, med, torch.full_like(med, complex(float("nan"), float("nan")))
            )
            return med

        # ---- Branch: cell_ids shape [N] (shared mapping) ----
        if cell_ids.ndim == 1:
            groups = (cell_ids // 4).to(torch.long)  # [N] parent ids (global)
            parents, inv = torch.unique(groups, sorted=True, return_inverse=True)
            n_bins = parents.numel()

            R = int(np.prod(orig)) if len(orig) > 0 else 1
            data_flat = data.reshape(R, N)  # [R, N]

            if reduce in ("mean", "max"):
                # Vectorized scatter path (same as before)
                row_offsets = torch.arange(R, device=data.device).unsqueeze(1) * n_bins
                idx = inv.unsqueeze(0).expand(R, -1) + row_offsets
                vals_flat = data_flat.reshape(-1)
                idx_flat = idx.reshape(-1)
                out_size = R * n_bins

                if reduce == "mean":
                    out_sum = torch.zeros(
                        out_size, dtype=data.dtype, device=data.device
                    )
                    out_cnt = torch.zeros_like(out_sum)
                    out_sum.scatter_add_(0, idx_flat, vals_flat)
                    out_cnt.scatter_add_(0, idx_flat, torch.ones_like(vals_flat))
                    out_cnt.clamp_(min=1)
                    reduced = out_sum / out_cnt
                else:  # "max"
                    reduced = _segment_max(vals_flat, idx_flat, out_size)

                output = reduced.view(*orig, n_bins)
                return output, parents

            elif reduce == "median":
                # Build a 4-slot array per parent using child offset = cell_ids % 4
                off = (cell_ids % 4).to(torch.long)  # [N] in {0,1,2,3}
                out4 = torch.full(
                    (R, n_bins, 4), torch.nan, dtype=data.dtype, device=data.device
                )
                # flat indexing to scatter
                base = torch.arange(R, device=data.device).unsqueeze(1) * (n_bins * 4)
                flat_index = base + (inv.unsqueeze(0) * 4) + off.unsqueeze(0)  # [R, N]
                out4 = out4.reshape(-1)
                out4.scatter_(0, flat_index.reshape(-1), data_flat.reshape(-1))
                out4 = out4.view(R, n_bins, 4)  # [R, n_bins, 4]

                if torch.is_complex(data):
                    med = _median_from_four_complex(out4)  # [R, n_bins]
                else:
                    med = _median_from_four_real(out4)  # [R, n_bins]

                output = med.view(*orig, n_bins)
                return output, parents
            else:
                raise ValueError("reduce must be 'mean', 'max', or 'median'.")

        # ---- Branch: cell_ids shape [B, N] (per-batch mapping) ----
        elif cell_ids.ndim == 2:
            B = cell_ids.shape[0]
            R = int(np.prod(orig)) if len(orig) > 0 else 1
            data_flat = data.reshape(R, N)
            B_data = data.shape[0] if len(orig) > 0 else 1
            if B_data % B != 0:
                raise ValueError(
                    f"Leading dim of data ({B_data}) must be a multiple of cell_ids batch ({B})."
                )
            T = (R // B_data) if B_data > 0 else 1  # repeats per batch row

            outs_list, groups_list = [], []
            max_bins = 0

            for b in range(B):
                groups_b = (cell_ids[b] // 4).to(torch.long)  # [N]
                parents_b, inv_b = torch.unique(
                    groups_b, sorted=True, return_inverse=True
                )
                n_bins_b = parents_b.numel()
                max_bins = max(max_bins, n_bins_b)

                # rows of data_flat that correspond to this batch row
                start, stop = b * T, (b + 1) * T
                rows = slice(start, stop)  # T rows -> [T, N]
                vals = data_flat[rows]

                if reduce in ("mean", "max"):
                    row_offsets = (
                        torch.arange(T, device=data.device).unsqueeze(1) * n_bins_b
                    )
                    idx = inv_b.unsqueeze(0).expand(T, -1) + row_offsets
                    vals_flat = vals.reshape(-1)
                    idx_flat = idx.reshape(-1)
                    out_size = T * n_bins_b
                    if reduce == "mean":
                        out_sum = torch.zeros(
                            out_size, dtype=data.dtype, device=data.device
                        )
                        out_cnt = torch.zeros_like(out_sum)
                        out_sum.scatter_add_(0, idx_flat, vals_flat)
                        out_cnt.scatter_add_(0, idx_flat, torch.ones_like(vals_flat))
                        out_cnt.clamp_(min=1)
                        reduced_bt = (out_sum / out_cnt).view(T, n_bins_b)
                    else:
                        reduced_bt = _segment_max(vals_flat, idx_flat, out_size).view(
                            T, n_bins_b
                        )
                    outs_list.append(reduced_bt)
                    groups_list.append(parents_b)
                elif reduce == "median":
                    off_b = (cell_ids[b] % 4).to(torch.long)  # [N] in {0,1,2,3}
                    out4 = torch.full(
                        (T, n_bins_b, 4),
                        torch.nan,
                        dtype=data.dtype,
                        device=data.device,
                    )
                    base = torch.arange(T, device=data.device).unsqueeze(1) * (
                        n_bins_b * 4
                    )
                    flat_index = (
                        base + (inv_b.unsqueeze(0) * 4) + off_b.unsqueeze(0)
                    )  # [T, N]
                    out4 = out4.reshape(-1)
                    out4.scatter_(0, flat_index.reshape(-1), vals.reshape(-1))
                    out4 = out4.view(T, n_bins_b, 4)  # [T, n_bins_b, 4]

                    if torch.is_complex(data):
                        reduced_bt = _median_from_four_complex(out4)  # [T, n_bins_b]
                    else:
                        reduced_bt = _median_from_four_real(out4)  # [T, n_bins_b]

                    outs_list.append(reduced_bt)
                    groups_list.append(parents_b)
                else:
                    raise ValueError("reduce must be 'mean', 'max', or 'median'.")

            if not padded:
                return outs_list, groups_list

            # Padded output (B, T, max_bins) [+ mask]
            out_pad = torch.full(
                (B, T, max_bins), fill_value, dtype=data.dtype, device=data.device
            )
            groups_pad = torch.full(
                (B, max_bins), -1, dtype=torch.long, device=data.device
            )
            mask = torch.zeros((B, max_bins), dtype=torch.bool, device=data.device)
            for b, (o_b, g_b) in enumerate(zip(outs_list, groups_list)):
                nb = g_b.numel()
                out_pad[b, :, :nb] = o_b
                groups_pad[b, :nb] = g_b
                mask[b, :nb] = True

            if len(orig) > 1:
                extra = orig[1:]
                out_pad = out_pad.view(B, *extra, max_bins)
            else:
                out_pad = out_pad.view(B, max_bins)

            return out_pad, groups_pad, mask

        else:
            raise ValueError("`cell_ids` must be of shape [N] or [B, N].")

        else:
            raise ValueError("`cell_ids` must be of shape [N] or [B, N].")
    
    def average_by_cell_group(data, cell_ids):
        """
        data: tensor of shape [..., N, ...] (ex: [B, N, C])
        cell_ids: tensor of shape [N]
        Returns: mean_data of shape [..., G, ...] where G = number of unique cell_ids//4
        """
        original_shape = data.shape
        leading = data.shape[:-2]  # all dims before N
        N = data.shape[-2]
        trailing = data.shape[-1:]  # all dims after N

        groups = (cell_ids // 4).long()  # [N]
        unique_groups, group_indices, counts = torch.unique(
            groups, return_inverse=True, return_counts=True
        )

        return torch.bincount(group_indices, weights=data) / counts, unique_groups

    def bk_masked_median(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        max_iter: int = 100,
        tol: float = 1e-6,
        eps: float = 1e-12,
    ):
        """
        Masked geometric median over the last axis using Weiszfeld iteration (1D case).

        Parameters
        ----------
        x : torch.Tensor
            Shape [a, b, c, N]. Can be real or complex.
        mask : torch.Tensor
            Binary mask of shape [a, b, 1, N]; broadcast across 'c'.
        max_iter : int
            Max number of Weiszfeld iterations.
        tol : float
            Convergence tolerance on the max absolute update per voxel.
        eps : float
            Small value to avoid division-by-zero in the weights.

        Returns
        -------
        med  : torch.Tensor, shape [a, b, c]
            Geometric median of x along the last axis where mask == 1.
            - For complex x: distances use the complex magnitude |x - y|.
              The returned median is complex.
        med2 : torch.Tensor, shape [a, b, c]
            Geometric median of squared values along the last axis where mask == 1.
            - If x is real : median of x**2 (real).
            - If x is complex : median of |x|**2 (real).
        """

        # --- helpers ---
        def _nan_like(y: torch.Tensor) -> torch.Tensor:
            """Return a NaN tensor with the same shape/dtype/device as y."""
            if torch.is_complex(y):
                return torch.full_like(y, complex(float("nan"), float("nan")))
            else:
                return torch.full_like(y, float("nan"))

        def safe_nanmax(t: torch.Tensor) -> torch.Tensor:
            """
            Backward-compatible replacement for torch.nanmax.
            Assumes t is real-valued (we only call it on absolute updates).
            """
            if torch.isnan(t).all():
                return torch.tensor(float("nan"), device=t.device, dtype=t.dtype)
            if torch.isnan(t).any():
                return torch.max(t[~torch.isnan(t)])
            return torch.max(t)

        # --- prep shapes & mask ---
        # Broadcast mask to x's shape [a,b,c,N]
        mask_bool = mask.to(torch.bool).expand_as(x)  # [a,b,c,N]
        m_float = mask_bool.to(dtype=x.real.dtype)  # weights need real dtype

        # Count valid samples per voxel
        valid_counts = mask_bool.sum(dim=-1)  # [a,b,c]
        zero_valid = valid_counts == 0

        # Denominator for masked mean initialization (avoid div-by-zero with clamp_min)
        denom = valid_counts.clamp_min(1).to(dtype=x.real.dtype)  # real

        # --- initialize y with masked mean (good starting point) ---
        if torch.is_complex(x):
            # (m_float*x) promotes to complex; denom to complex for division
            y = (m_float * x).sum(dim=-1) / denom.to(dtype=x.dtype)  # [a,b,c], complex
        else:
            y = (m_float * x).sum(dim=-1) / denom  # [a,b,c], real

        # Put NaNs where there are no valid samples
        y = torch.where(zero_valid, _nan_like(y), y)

        # --- Weiszfeld iterations for x -> med ---
        # y_{k+1} = sum_i (x_i / ||x_i - y_k||) / sum_i (1/||x_i - y_k||), masked
        for _ in range(max_iter):
            if torch.all(zero_valid):
                break

            diff = x - y.unsqueeze(-1)  # [a,b,c,N]
            dist = diff.abs()  # real, [a,b,c,N]
            w = m_float * (1.0 / torch.clamp(dist, min=eps))  # real weights
            w_sum = w.sum(dim=-1)  # [a,b,c], real
            y_new = (w * x).sum(dim=-1) / w_sum.clamp_min(eps)  # [a,b,c], real/complex

            # Keep NaNs on zero-valid voxels
            y_new = torch.where(zero_valid, _nan_like(y_new), y_new)

            # Convergence
            upd = (y_new - y).abs()  # real
            if safe_nanmax(upd).item() <= tol:
                y = y_new
                break
            y = y_new

        med = y  # [a,b,c]

        # --- Weiszfeld iterations for squared values -> med2 ---
        # For complex: use |x|^2; for real: x^2
        s = (x.abs() ** 2) if torch.is_complex(x) else (x**2)  # [a,b,c,N], real
        # Init with masked mean of s
        z = (m_float * s).sum(dim=-1) / denom  # [a,b,c], real
        z = torch.where(zero_valid, _nan_like(z), z)

        # Weiszfeld on real scalars s
        for _ in range(max_iter):
            if torch.all(zero_valid):
                break
            diff_s = s - z.unsqueeze(-1)  # [a,b,c,N]
            dist_s = diff_s.abs().clamp_min(eps)  # real
            w_s = m_float * (1.0 / dist_s)  # real
            w_s_sum = w_s.sum(dim=-1)  # [a,b,c]
            z_new = (w_s * s).sum(dim=-1) / w_s_sum.clamp_min(eps)  # [a,b,c]
            z_new = torch.where(zero_valid, _nan_like(z_new), z_new)

            upd_s = (z_new - z).abs()
            if safe_nanmax(upd_s).item() <= tol:
                z = z_new
                break
            z = z_new

        med2 = z  # [a,b,c], real

        return med, med2

    def bk_masked_median_2d_weiszfeld(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        max_iter: int = 100,
        tol: float = 1e-6,
        eps: float = 1e-12,
    ):
        """
        Masked geometric median over 2D spatial axes using Weiszfeld iteration.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape [a, b, c, N1, N2]. Can be real or complex.
        mask : torch.Tensor
            Binary mask of shape [a, b, 1, N1, N2]; broadcasted over 'c'.
        max_iter : int
            Maximum number of Weiszfeld iterations.
        tol : float
            Stopping tolerance on the max absolute update per voxel.
        eps : float
            Small positive value to avoid division by zero in weights.

        Returns
        -------
        med : torch.Tensor, shape [a, b, c]
            Geometric median of x over (N1, N2) where mask == 1.
            - If x is complex, distances are magnitudes |x - y| in the complex plane,
              and the returned value is the complex sample estimate (not its magnitude).
        med2 : torch.Tensor, shape [a, b, c]
            Geometric median of squared values over (N1, N2) where mask == 1.
            - If x is real : median of x**2 (via Weiszfeld in 1D).
            - If x is complex : median of |x|**2 (real, non-negative).

        Notes
        -----
        - Voxels with zero valid samples return NaN (NaN+NaNj for complex med).
        - Weiszfeld update: y_{k+1} = sum_i w_i x_i / sum_i w_i with w_i = 1 / ||x_i - y_k||.
          Here ||.|| is |.| for real numbers and the complex magnitude for complex numbers.
        """
        # Broadcast mask to x's shape [a,b,c,N1,N2] and flatten spatial dims to N
        mask_bool = mask.to(torch.bool).expand_as(x)
        a, b, c, N1, N2 = x.shape
        N = N1 * N2
        x_flat = x.reshape(a, b, c, N)
        m_flat = mask_bool.reshape(a, b, c, N)

        # Count valid samples per voxel
        valid_counts = m_flat.sum(dim=-1)  # [a,b,c]

        # Helper to create NaN of the right dtype
        def _nan_like(y):
            if torch.is_complex(y):
                return torch.full_like(y, complex(float("nan"), float("nan")))
            else:
                return torch.full_like(y, float("nan"))

        # --- Geometric median of x (real or complex) ---
        # Initialize y0: masked mean (robust enough as a starting point)
        # y0 = sum(mask * x) / sum(mask)
        denom = valid_counts.clamp_min(1).to(x.dtype)
        if torch.is_complex(x):
            denom_c = denom.to(x.dtype)
            y = (m_flat * x_flat).sum(dim=-1) / denom_c  # [a,b,c]
        else:
            y = (m_flat * x_flat).sum(dim=-1) / denom  # [a,b,c]

        # Where there are zero valid samples, set to NaN now (and keep NaN through)
        zero_valid = valid_counts == 0
        if torch.is_complex(x):
            y = torch.where(
                zero_valid, torch.full_like(y, complex(float("nan"), float("nan"))), y
            )
        else:
            y = torch.where(zero_valid, torch.full_like(y, float("nan")), y)

        # helper: nanmax replacement
        def safe_nanmax(t):
            if torch.isnan(t).all():
                return torch.tensor(float("nan"), device=t.device, dtype=torch.float32)
            return (
                torch.max(t[~torch.isnan(t)]) if torch.isnan(t).any() else torch.max(t)
            )

        # Iterate Weiszfeld
        for _ in range(max_iter):
            # Skip voxels with no valid samples
            if torch.all(zero_valid):
                break

            # diff: [a,b,c,N], distances are |diff|
            diff = x_flat - y.unsqueeze(-1)  # broadcast y over N
            dist = diff.abs()  # real tensor, [a,b,c,N]

            # weights w = mask / max(dist, eps)
            w = m_flat * (1.0 / torch.clamp(dist, min=eps))  # [a,b,c,N]
            w_sum = w.sum(dim=-1)  # [a,b,c], real

            # Next iterate y_new = sum(w * x) / sum(w)
            # For complex x, w is real so (w*x) is complex — OK.
            y_new = (w * x_flat).sum(dim=-1) / w_sum.clamp_min(eps)

            # Keep NaN on zero-valid voxels
            if torch.is_complex(x):
                y_new = torch.where(
                    zero_valid,
                    torch.full_like(y_new, complex(float("nan"), float("nan"))),
                    y_new,
                )
            else:
                y_new = torch.where(
                    zero_valid, torch.full_like(y_new, float("nan")), y_new
                )

            # Convergence check (max absolute update over all voxels)
            upd = (y_new - y).abs()
            if safe_nanmax(upd).item() <= tol:
                y = y_new
                break
            y = y_new

        med = y  # [a,b,c]

        # --- Geometric median of squared values (med2) ---
        if torch.is_complex(x):
            s_flat = x_flat.abs() ** 2  # [a,b,c,N], real
        else:
            s_flat = x_flat**2  # [a,b,c,N], real

        # Initialize z0 = masked mean of s
        z = (m_flat * s_flat).sum(dim=-1) / denom  # [a,b,c], real
        z = torch.where(zero_valid, torch.full_like(z, float("nan")), z)

        # Weiszfeld on scalars (1D) for s: distances are |s_i - z|
        for _ in range(max_iter):
            if torch.all(zero_valid):
                break
            diff_s = s_flat - z.unsqueeze(-1)  # [a,b,c,N]
            dist_s = diff_s.abs().clamp_min(eps)  # avoid div-by-zero
            w_s = m_flat * (1.0 / dist_s)
            w_s_sum = w_s.sum(dim=-1)
            z_new = (w_s * s_flat).sum(dim=-1) / w_s_sum.clamp_min(eps)
            z_new = torch.where(zero_valid, torch.full_like(z_new, float("nan")), z_new)
            upd_s = (z_new - z).abs()
            if safe_nanmax(upd_s).item() <= tol:
                z = z_new
                break
            z = z_new

        med2 = z  # [a,b,c], real

        return med, med2

    # ---------------------------------------------−---------
    # --             BACKEND DEFINITION                    --
    # ---------------------------------------------−---------
    def bk_len(self, S):
        if S is None:
            return 0
        return S.numel()

    def bk_SparseTensor(self, indice, w, dense_shape=[]):
        return (
            self.backend.sparse_coo_tensor(indice, w, dense_shape)
            .coalesce()
            .to_sparse_csr()
            .to(self.torch_device)
        )

    def bk_stack(self, list, axis=0):
        return self.backend.stack(list, axis=axis).to(self.torch_device)

    def bk_sparse_dense_matmul(self, smat, mat):
        return smat.matmul(mat)

    def conv2d(self, x, w):
        """
        Perform 2D convolution using PyTorch format.

        Args:
            x: Tensor of shape [..., Nx, Ny] – input
            w: Tensor of shape [O_c, wx, wy] – conv weights

        Returns:
            Tensor of shape [..., O_c, Nx, Ny]
        """
        *leading_dims, Nx, Ny = x.shape  # extract leading dims
        O_c, wx, wy = w.shape

        # Flatten leading dims into batch dimension
        B = int(torch.prod(torch.tensor(leading_dims))) if leading_dims else 1
        x = x.reshape(B, 1, Nx, Ny)  # [B, 1, Nx, Ny]

        # Reshape filters to match conv2d format [O_c, 1, wx, wy]
        w = w[:, None, :, :]  # [O_c, 1, wx, wy]

        pad_x = wx // 2
        pad_y = wy // 2

        # Reflective padding to reduce edge artifacts
        x_padded = F.pad(x, (pad_y, pad_y, pad_x, pad_x), mode="reflect")

        # Apply convolution
        y = F.conv2d(x_padded, w)  # [B, O_c, Nx, Ny]

        # Restore original leading dimensions
        y = y.reshape(*leading_dims, O_c, Nx, Ny)

        return y
    
    def conv1d(self, x, w, strides=[1, 1, 1], padding="SAME"):
        """
        Performs 1D convolution along the last axis of a 2D tensor x[n, m] with kernel w[K].

        Parameters:
        - x: torch.Tensor of shape [n, m]
        - w: torch.Tensor of shape [K]
        - strides: list of 3 ints; only strides[1] (along axis -1) is used
        - padding: "SAME" or "VALID"

        Returns:
        - torch.Tensor of shape [n, m] (if SAME) or smaller (if VALID)
        """
        assert x.ndim == 2, "Input x must be a 2D tensor [n, m]"
        assert w.ndim == 1, "Kernel w must be a 1D tensor [K]"
        stride = strides[1]

        # Reshape for PyTorch conv1d: [batch, channels, width]
        x_reshaped = x.unsqueeze(1)  # [n, 1, m]
        w_flipped = w.flip(0).view(
            1, 1, -1
        )  # [out_channels=1, in_channels=1, kernel_size]

        if padding.upper() == "SAME":
            pad_total = w.shape[0] - 1
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            x_reshaped = F.pad(
                x_reshaped, (pad_left, pad_right), mode="constant", value=0
            )
            padding_mode = "valid"
        elif padding.upper() == "VALID":
            padding_mode = "valid"
        else:
            raise ValueError("padding must be either 'SAME' or 'VALID'")

        out = F.conv1d(
            x_reshaped, w_flipped, stride=stride, padding=0
        )  # manual padding applied above
        return out.squeeze(1)  # [n, m_out]

    def bk_threshold(self, x, threshold, greater=True):

        x.to(x.dtype)
        return (x > threshold) * x

    def bk_maximum(self, x1, x2):
        return self.backend.maximum(x1, x2)

    def bk_device(self, device_name):
        return self.backend.device(device_name)

    def bk_ones(self, shape, dtype=None):
        if dtype is None:
            dtype = self.all_type
        return self.bk_cast(np.ones(shape))

    def bk_conv1d(self, x, w):
        # Torch not yet done !!!
        return self.backend.nn.conv1d(x, w, stride=1, padding="SAME")

    def bk_flattenR(self, x):
        if self.bk_is_complex(x):
            rr = self.backend.reshape(
                self.bk_real(x), [np.prod(np.array(list(x.shape)))]
            )
            ii = self.backend.reshape(
                self.bk_imag(x), [np.prod(np.array(list(x.shape)))]
            )
            return self.bk_concat([rr, ii], axis=0)
        else:
            return self.backend.reshape(x, [np.prod(np.array(list(x.shape)))])

    def bk_flatten(self, x):
        return self.backend.reshape(x, [np.prod(np.array(list(x.shape)))])

    def bk_resize_image(self, x, shape):
        tmp = self.backend.nn.functional.interpolate(
            x.permute(0, 3, 1, 2), size=shape, mode="bilinear", align_corners=False
        )
        return self.bk_cast(tmp.permute(0, 2, 3, 1))

    def bk_L1(self, x):
        if x.dtype == self.all_cbk_type:
            xr = self.bk_real(x)
            # xi = self.bk_imag(x)

            r = self.backend.sign(xr) * self.backend.sqrt(
                self.backend.sign(xr) * xr + 1e-16
            )
            # return r
            # i = self.backend.sign(xi) * self.backend.sqrt(self.backend.sign(xi) * xi)

            return r
        else:
            return self.backend.sign(x) * self.backend.sqrt(
                self.backend.sign(x) * x + 1e-16
            )

    def bk_square_comp(self, x):
        if x.dtype == self.all_cbk_type:
            xr = self.bk_real(x)
            xi = self.bk_imag(x)

            r = xr * xr
            i = xi * xi
            return self.bk_complex(r, i)
        else:
            return x * x

    def bk_reduce_sum(self, data, axis=None):

        if axis is None:
            return self.backend.sum(data)
        else:
            return self.backend.sum(data, axis)

    # ---------------------------------------------−---------
    # return a tensor size

    def bk_size(self, data):
        return data.numel()

    def constant(self, data):
        return data

    def bk_reduce_mean(self, data, axis=None):

        if axis is None:
            return self.backend.mean(data)
        else:
            return self.backend.mean(data, axis)
        
    def bk_reduce_median(self, data, axis=None):

        if axis is None:
            res,_ = self.backend.median(data)
        else:
            res,_ = self.backend.median(data, axis)
        return res

    def bk_reduce_median(self, data, axis=None):

        if axis is None:
            res, _ = self.backend.median(data)
        else:
            res, _ = self.backend.median(data, axis)
        return res

    def bk_reduce_min(self, data, axis=None):

        if axis is None:
            return self.backend.min(data)
        else:
            return self.backend.min(data, axis)

    def bk_random_seed(self, value):

        return self.backend.random.set_seed(value)

    def bk_random_uniform(self, shape):

        return self.backend.random.uniform(shape)

    def bk_reduce_std(self, data, axis=None):
        if axis is None:
            r = self.backend.std(data)
        else:
            r = self.backend.std(data, axis)

        if self.bk_is_complex(data):
            return self.bk_complex(r, 0 * r)
        else:
            return r

    def bk_sqrt(self, data):

        return self.backend.sqrt(self.backend.abs(data))

    def bk_abs(self, data):
        return self.backend.abs(data)

    def bk_is_complex(self, data):

        if isinstance(data, np.ndarray):
            return data.dtype == "complex64" or data.dtype == "complex128"

        return data.dtype.is_complex

    def bk_distcomp(self, data):
        if self.bk_is_complex(data):
            res = self.bk_square(self.bk_real(data)) + self.bk_square(
                self.bk_imag(data)
            )
            return res
        else:
            return self.bk_square(data)

    def bk_norm(self, data):
        if self.bk_is_complex(data):
            res = self.bk_square(self.bk_real(data)) + self.bk_square(
                self.bk_imag(data)
            )
            return self.bk_sqrt(res)

        else:
            return self.bk_abs(data)

    def bk_square(self, data):

        return self.backend.square(data)

    def bk_log(self, data):
        return self.backend.log(data)

    def bk_matmul(self, a, b):
        return self.backend.matmul(a, b)

    def bk_tensor(self, data):
        return self.backend.constant(data).to(self.torch_device)

    def bk_shape_tensor(self, shape):
        return self.backend.tensor(shape=shape).to(self.torch_device)

    def bk_complex(self, real, imag):
        return self.backend.complex(real, imag).to(self.torch_device)

    def bk_exp(self, data):

        return self.backend.exp(data)

    def bk_min(self, data):

        return self.backend.reduce_min(data)

    def bk_argmin(self, data):

        return self.backend.argmin(data)

    def bk_tanh(self, data):

        return self.backend.math.tanh(data)

    def bk_max(self, data):

        return self.backend.reduce_max(data)

    def bk_argmax(self, data):

        return self.backend.argmax(data)

    def bk_reshape(self, data, shape):
        # if isinstance(data, np.ndarray):
        #    return data.reshape(shape)
        return data.reshape(shape)

    def bk_repeat(self, data, nn, axis=0):
        return self.backend.repeat_interleave(data, repeats=nn, dim=axis)

    def bk_tile(self, data, nn, axis=0):

        return self.backend.tile(data, dims=[nn])

    def bk_roll(self, data, nn, axis=0):
        return self.backend.roll(data, nn, axis=axis)

    def bk_expand_dims(self, data, axis=0):
        if isinstance(data, np.ndarray):
            data = self.backend.from_numpy(data)
        return self.backend.unsqueeze(data, axis)

    def bk_transpose(self, data, thelist):
        return self.backend.transpose(data, thelist[0], thelist[1])

    def bk_concat(self, data, axis=None):

        if axis is None:
            if data[0].dtype == self.all_cbk_type:
                ndata = len(data)
                xr = self.backend.concat([self.bk_real(data[k]) for k in range(ndata)])
                xi = self.backend.concat([self.bk_imag(data[k]) for k in range(ndata)])
                return self.bk_complex(xr, xi)
            else:
                return self.backend.concat(data)
        else:
            if data[0].dtype == self.all_cbk_type:
                ndata = len(data)
                xr = self.backend.concat(
                    [self.bk_real(data[k]) for k in range(ndata)], axis=axis
                )

                xi = self.backend.concat(
                    [self.bk_imag(data[k]) for k in range(ndata)], axis=axis
                )
                return self.bk_complex(xr, xi)
            else:
                return self.backend.concat(data, axis=axis)

    def bk_zeros(self, shape, dtype=None):
        return self.backend.zeros(shape, dtype=dtype).to(self.torch_device)

    def bk_gather(self, data, idx, axis=0):
        if axis == -1:
            return data[..., idx]
        elif axis == 0:
            return data[idx]
        elif axis == 1:
            return data[:, idx]
        elif axis == 2:
            return data[:, :, idx]
        elif axis == 3:
            return data[:, :, :, idx]
        return data[idx, ...]

    def bk_reverse(self, data, axis=0):
        return self.backend.flip(data, dims=[axis])

    def bk_fft(self, data):
        return self.backend.fft.fft(data)

    def bk_fftn(self, data, dim=None):
        return self.backend.fft.fftn(data, dim=dim)

    def bk_ifftn(self, data, dim=None, norm=None):
        return self.backend.fft.ifftn(data, dim=dim, norm=norm)

    def bk_rfft(self, data):
        return self.backend.fft.rfft(data)

    def bk_irfft(self, data):
        return self.backend.fft.irfft(data)

    def bk_conjugate(self, data):

        return self.backend.conj(data)

    def bk_real(self, data):
        return data.real

    def bk_imag(self, data):
        if data.dtype == self.all_cbk_type:
            return data.imag
        else:
            return 0

    def bk_relu(self, x):
        return self.backend.relu(x)

    def bk_clip_by_value(self, x, xmin, xmax):
        if isinstance(x, np.ndarray):
            x = np.clip(x, xmin, xmax)
        x = (
            self.backend.tensor(x, dtype=self.backend.float32)
            if not isinstance(x, self.backend.Tensor)
            else x
        )
        xmin = (
            self.backend.tensor(xmin, dtype=self.backend.float32)
            if not isinstance(xmin, self.backend.Tensor)
            else xmin
        )
        xmax = (
            self.backend.tensor(xmax, dtype=self.backend.float32)
            if not isinstance(xmax, self.backend.Tensor)
            else xmax
        )
        return self.backend.clamp(x, min=xmin, max=xmax)

    def bk_cast(self, x):
        if isinstance(x, np.float64):
            if self.all_bk_type == "float32":
                return self.backend.tensor(np.float32(x)).to(self.torch_device)
            else:
                return self.backend.tensor(x).to(self.torch_device)
        if isinstance(x, np.float32):
            if self.all_bk_type == "float64":
                return self.backend.tensor(np.float64(x)).to(self.torch_device)
            else:
                return self.backend.tensor(x).to(self.torch_device)
        if isinstance(x, np.complex128):
            if self.all_bk_type == "float32":
                return self.backend.tensor(np.complex64(x)).to(self.torch_device)
            else:
                return self.backend.tensor(x).to(self.torch_device)
        if isinstance(x, np.complex64):
            if self.all_bk_type == "float64":
                return self.backend.tensor(np.complex128(x)).to(self.torch_device)
            else:
                return self.backend.tensor(x).to(self.torch_device)

        if isinstance(x, np.int32) or isinstance(x, np.int64) or isinstance(x, int):
            if self.all_bk_type == "float64":
                return self.backend.tensor(np.float64(x)).to(self.torch_device)
            else:
                return self.backend.tensor(np.float32(x)).to(self.torch_device)

        if self.bk_is_complex(x):
            out_type = self.all_cbk_type
        else:
            out_type = self.all_bk_type

        if isinstance(x, np.ndarray):
            x = self.backend.from_numpy(x).to(self.torch_device)

        if x.dtype.is_complex:
            out_type = self.all_cbk_type
        else:
            out_type = self.all_bk_type

        return x.type(out_type).to(self.torch_device)

    def bk_variable(self, x):
        return self.bk_cast(x)

    def bk_assign(self, x, y):
        return y

    def bk_constant(self, x):

        return self.bk_cast(x)

    def bk_cos(self, x):
        return self.backend.cos(x)

    def bk_sin(self, x):
        return self.backend.sin(x)

    def bk_arctan2(self, c, s):
        return self.backend.arctan2(c, s)

    def bk_empty(self, list):
        return self.backend.empty(list)

    def to_numpy(self, x):
        if isinstance(x, np.ndarray):
            return x

        return x.cpu().numpy()
