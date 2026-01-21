import torch
import torch.nn as nn
import numpy as np

from foscat.SphereDownGeo import SphereDownGeo


class SphereUpGeo(nn.Module):
    """Geometric HEALPix upsampling operator using the transpose of SphereDownGeo.

    `cell_ids_out` (coarse pixels at nside_out, NESTED) is mandatory.
    Forward expects x of shape [B, C, K_out] aligned with that order.
    Output is a full fine-grid map [B, C, N_in] at nside_in = 2*nside_out.

    Normalization (diagonal corrections):
      - up_norm='adjoint': x_up = M^T x
      - up_norm='col_l1':  x_up = (M^T x) / col_sum, col_sum[i] = sum_k M[k,i]
      - up_norm='diag_l2': x_up = (M^T x) / col_l2,  col_l2[i]  = sum_k M[k,i]^2
    """

    def __init__(
        self,
        nside_out: int,
        cell_ids_out,
        radius_deg: float | None = None,
        sigma_deg: float | None = None,
        weight_norm: str = "l1",
        up_norm: str = "col_l1",
        eps: float = 1e-12,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()

        if cell_ids_out is None:
            raise ValueError("cell_ids_out is mandatory (1D list/np/tensor of coarse HEALPix ids at nside_out).")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.dtype = dtype

        self.nside_out = int(nside_out)
        assert (self.nside_out & (self.nside_out - 1)) == 0, "nside_out must be a power of 2."
        self.nside_in = self.nside_out * 2

        self.N_out = 12 * self.nside_out * self.nside_out
        self.N_in = 12 * self.nside_in * self.nside_in

        up_norm = str(up_norm).lower().strip()
        if up_norm not in ("adjoint", "col_l1", "diag_l2"):
            raise ValueError("up_norm must be 'adjoint', 'col_l1', or 'diag_l2'.")
        self.up_norm = up_norm
        self.eps = float(eps)

        # Coarse ids in user-provided order (must be unique for alignment)
        if isinstance(cell_ids_out, torch.Tensor):
            cell_ids_out_np = cell_ids_out.detach().cpu().numpy().astype(np.int64)
        else:
            cell_ids_out_np = np.asarray(cell_ids_out, dtype=np.int64)

        if cell_ids_out_np.ndim != 1:
            raise ValueError("cell_ids_out must be 1D")
        if cell_ids_out_np.size == 0:
            raise ValueError("cell_ids_out must be non-empty")
        if cell_ids_out_np.min() < 0 or cell_ids_out_np.max() >= self.N_out:
            raise ValueError("cell_ids_out contains out-of-bounds ids for this nside_out")
        if np.unique(cell_ids_out_np).size != cell_ids_out_np.size:
            raise ValueError("cell_ids_out must not contain duplicates (order matters for alignment).")

        self.cell_ids_out_np = cell_ids_out_np
        self.K_out = int(cell_ids_out_np.size)
        self.register_buffer("cell_ids_out_t", torch.as_tensor(cell_ids_out_np, dtype=torch.long, device=self.device))

        # Build the FULL down operator at fine resolution (nside_in -> nside_out)
        tmp_down = SphereDownGeo(
            nside_in=self.nside_in,
            mode="smooth",
            radius_deg=radius_deg,
            sigma_deg=sigma_deg,
            weight_norm=weight_norm,
            device=self.device,
            dtype=self.dtype,
        )

        M_down_full = torch.sparse_coo_tensor(
            tmp_down.M_indices,
            tmp_down.M_values,
            size=(tmp_down.N_out, tmp_down.N_in),
            device=self.device,
            dtype=self.dtype,
        ).coalesce()

        # Extract ONLY the requested coarse rows, in the provided order.
        # We do this on CPU with numpy for simplicity and speed at init.
        idx = M_down_full.indices().cpu().numpy()
        vals = M_down_full.values().cpu().numpy()
        rows = idx[0]
        cols = idx[1]

        # Map original row id -> new row position [0..K_out-1]
        row_map = {int(r): i for i, r in enumerate(cell_ids_out_np.tolist())}
        mask = np.fromiter((r in row_map for r in rows), dtype=bool, count=rows.size)

        rows_sel = rows[mask]
        cols_sel = cols[mask]
        vals_sel = vals[mask]

        new_rows = np.fromiter((row_map[int(r)] for r in rows_sel), dtype=np.int64, count=rows_sel.size)

        M_down_sub = torch.sparse_coo_tensor(
            torch.as_tensor(np.stack([new_rows, cols_sel], axis=0), dtype=torch.long),
            torch.as_tensor(vals_sel, dtype=self.dtype),
            size=(self.K_out, self.N_in),
            device=self.device,
            dtype=self.dtype,
        ).coalesce()

        # Store M^T (sparse) so forward is just sparse.mm
        M_up = self._transpose_sparse(M_down_sub)  # [N_in, K_out]
        self.register_buffer("M_indices", M_up.indices())
        self.register_buffer("M_values", M_up.values())
        self.M_size = M_up.size()

        # Diagonal normalizers (length N_in), based on the selected coarse rows only
        idx_sub = M_down_sub.indices()
        vals_sub = M_down_sub.values()
        fine_cols = idx_sub[1]

        col_sum = torch.zeros(self.N_in, device=self.device, dtype=self.dtype)
        col_l2 = torch.zeros(self.N_in, device=self.device, dtype=self.dtype)
        col_sum.scatter_add_(0, fine_cols, vals_sub)
        col_l2.scatter_add_(0, fine_cols, vals_sub * vals_sub)

        self.register_buffer("col_sum", col_sum)
        self.register_buffer("col_l2", col_l2)

        # Fine ids (full sphere)
        self.register_buffer("cell_ids_in_t", torch.arange(self.N_in, dtype=torch.long, device=self.device))

        self.M_T =  torch.sparse_coo_tensor(
            self.M_indices.to(device=self.device),
            self.M_values.to(device=self.device, dtype=self.dtype),
            size=self.M_size,
            device=self.device,
            dtype=self.dtype,
        ).coalesce() #.to_sparse_csr().to(self.device)
        
    @staticmethod
    def _transpose_sparse(M: torch.Tensor) -> torch.Tensor:
        M = M.coalesce()
        idx = M.indices()
        vals = M.values()
        R, C = M.size()
        idx_T = torch.stack([idx[1], idx[0]], dim=0)
        return torch.sparse_coo_tensor(idx_T, vals, size=(C, R), device=M.device, dtype=M.dtype).coalesce()

    def forward(self, x: torch.Tensor):
        """x: [B, C, K_out] -> x_up: [B, C, N_in]."""
        B, C, K_out = x.shape
        assert K_out == self.K_out, f"Expected K_out={self.K_out}, got {K_out}"

        x_bc = x.reshape(B * C, K_out)
        x_up_bc_T = torch.sparse.mm(self.M_T, x_bc.T)    # [N_in, B*C]
        x_up = x_up_bc_T.T.reshape(B, C, self.N_in) # [B, C, N_in]

        if self.up_norm == "col_l1":
            denom = self.col_sum.to(device=x.device, dtype=x.dtype).clamp_min(self.eps)
            x_up = x_up / denom.view(1, 1, -1)
        elif self.up_norm == "diag_l2":
            denom = self.col_l2.to(device=x.device, dtype=x.dtype).clamp_min(self.eps)
            x_up = x_up / denom.view(1, 1, -1)

        return x_up, self.cell_ids_in_t.to(device=x.device)
