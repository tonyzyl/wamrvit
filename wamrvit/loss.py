import torch
import torch.nn.functional as F


class LpLoss:
    # loss function with rel/abs Lp loss, modified from neuralop:
    # https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/losses/data_losses.py
    """
    LpLoss: Lp loss function, return the relative loss by default
    Args:
        d: int, start dimension of the field. E,g., for shape like (b, c, h, w), d=2 (default 1)
        p: int, p in Lp norm, default 2
        reduce_dims: int or list of int, dimensions to reduce
        reductions: str or list of str, 'sum' or 'mean'
        eps: float, clamping the denominator in relative loss.
        method: 'rel' or 'abs', return relative or absolute loss

    Call: (y_pred, y, weight=None, mask=None)
    """

    def __init__(self, d=1, p=2, reduce_dims=0, reductions="mean", method="rel", eps=1e-6):
        super().__init__()

        self.d = d
        self.p = p
        self.eps = eps
        if method == "rel":
            self.method = self.rel
        elif method == "abs":
            self.method = self.abs
        else:
            raise ValueError("method should be 'rel' or 'abs'")

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims

        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions == "sum" or reductions == "mean"
                self.reductions = [reductions] * len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == "sum" or reductions[j] == "mean"
                self.reductions = reductions

    def reduce_all(self, x):
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == "sum":
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)
        return x

    def abs(self, x, y, weight: torch.Tensor | None = None):
        if self.p == 2:
            # Numerically stable L2 norm: sqrt(sum(x^2) + eps)
            if weight is None:
                diff_sq = torch.sum(
                    (torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d))
                    ** 2,
                    dim=-1,
                    keepdim=False,
                )
            else:
                diff_sq = torch.sum(
                    (torch.flatten(weight * (x - y), start_dim=-self.d)) ** 2,
                    dim=-1,
                    keepdim=False,
                )
            return torch.sqrt(diff_sq + self.eps)
        else:
            # Fallback for other p-norms (e.g., p=1)
            if weight is None:
                diff = torch.norm(
                    torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d),
                    p=self.p,
                    dim=-1,
                    keepdim=False,
                )
            else:
                diff = torch.norm(
                    torch.flatten(weight * (x - y), start_dim=-self.d),
                    p=self.p,
                    dim=-1,
                    keepdim=False,
                )
            return diff

    def rel(self, x, y, weight: torch.Tensor | None = None):
        if self.p == 2:
            # Numerically stable L2 norm: sqrt(sum(x^2) + eps)
            if weight is None:
                diff_sq = torch.sum(
                    (torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d))
                    ** 2,
                    dim=-1,
                    keepdim=False,
                )
                ynorm_sq = torch.sum(
                    torch.flatten(y, start_dim=-self.d) ** 2, dim=-1, keepdim=False
                )
            else:
                diff_sq = torch.sum(
                    (torch.flatten(weight * (x - y), start_dim=-self.d)) ** 2,
                    dim=-1,
                    keepdim=False,
                )
                ynorm_sq = torch.sum(
                    (torch.flatten(weight * y, start_dim=-self.d)) ** 2, dim=-1, keepdim=False
                )

            diff = torch.sqrt(diff_sq + self.eps)
            ynorm = torch.sqrt(ynorm_sq + self.eps)
        else:
            # Fallback for other p-norms (e.g., p=1)
            if weight is None:
                diff = torch.norm(
                    torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d),
                    p=self.p,
                    dim=-1,
                    keepdim=False,
                )
                ynorm = torch.norm(
                    torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False
                )
            else:
                diff = torch.norm(
                    torch.flatten(weight * (x - y), start_dim=-self.d),
                    p=self.p,
                    dim=-1,
                    keepdim=False,
                )
                ynorm = torch.norm(
                    torch.flatten(weight * y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False
                )

        ynorm = torch.clamp(ynorm, min=self.eps)  # Extra safety against division by zero
        diff = diff / ynorm  # (B, C)
        return diff

    def __call__(
        self, y_pred, y, weight: torch.Tensor | None = None, mask: torch.Tensor | None = None
    ):
        y = y.float()
        y_pred = y_pred.float()
        diff = self.method(y_pred, y, weight=weight)

        if mask is not None:
            # Broadcast mask (N,) to match diff shape (N, C, ...)
            mask_view_shape = [mask.shape[0]] + [1] * (diff.dim() - 1)
            mask_view = mask.view(*mask_view_shape)
            diff = diff * mask_view

        if self.reduce_dims is not None:
            # If masked, a standard mean over dim=0 divides by N instead of mask.sum()
            # We override the reduction logic for dim=0 if a mask is present
            if mask is not None and 0 in self.reduce_dims:
                for j in range(len(self.reduce_dims)):
                    dim = self.reduce_dims[j]
                    if self.reductions[j] == "sum":
                        diff = torch.sum(diff, dim=dim, keepdim=True)
                    else:  # mean
                        if dim == 0:
                            # Use sum / unmasked_count
                            diff = torch.sum(diff, dim=dim, keepdim=True) / torch.clamp(
                                mask.sum(), min=1e-8
                            )
                        else:
                            diff = torch.mean(diff, dim=dim, keepdim=True)
            else:
                diff = self.reduce_all(diff)

            diff = diff.squeeze()

        return diff

    @torch.no_grad()
    def get_loss_per_var(
        self,
        y_pred,
        y,
        weight: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        """Assuming input of order [atm_vars, sur_vars]"""
        diff = self.method(y_pred, y, weight=weight)
        y = y.float()
        y_pred = y_pred.float()

        if mask is not None:
            mask_view_shape = [mask.shape[0]] + [1] * (diff.dim() - 1)
            mask_view = mask.view(*mask_view_shape)
            diff = diff * mask_view

        batch_mean = torch.empty(
            y_pred.shape[1],
            device=y_pred.device,
            dtype=y_pred.dtype,
        )
        for i in range(0, y_pred.shape[1]):
            if mask is not None:
                # Divide by unmasked count for the mean
                batch_mean[i] = diff[:, i].sum() / torch.clamp(mask.sum(), min=1e-8)
            else:
                batch_mean[i] = diff[:, i].mean()

            if batch_mean[i] < 0:
                print(f"Negative mean for var {i}: {batch_mean[i]}")

        return batch_mean


class MSELoss:
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, y_pred, y, mask: torch.Tensor | None = None):
        # Always compute unreduced loss first so we can apply the mask
        diff = F.mse_loss(y_pred.float(), y.float(), reduction="none")

        if mask is not None:
            # Broadcast mask (N,) to match diff shape (N, C, ...)
            mask_view_shape = [mask.shape[0]] + [1] * (diff.dim() - 1)
            mask_view = mask.view(*mask_view_shape)
            diff = diff * mask_view

            if self.reduction == "mean":
                # Divide by (number of active patches * elements per patch)
                elements_per_patch = diff.numel() // diff.shape[0]
                return diff.sum() / (mask.sum() * elements_per_patch + 1e-8)
            elif self.reduction == "sum":
                return diff.sum()
            else:
                return diff
        else:
            # Standard PyTorch behavior if no mask is provided
            if self.reduction == "mean":
                return diff.mean()
            elif self.reduction == "sum":
                return diff.sum()
            else:
                return diff

    @torch.no_grad()
    def get_loss_per_var(
        self,
        y_pred,
        y,
        weight: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        """Assuming input of order [atm_vars, sur_vars]"""
        if weight is None:
            diff = (y_pred - y) ** 2
        else:
            diff = weight * ((y_pred - y) ** 2)

        if mask is not None:
            mask_view_shape = [mask.shape[0]] + [1] * (diff.dim() - 1)
            mask_view = mask.view(*mask_view_shape)
            diff = diff * mask_view

        batch_mean = torch.empty(
            y_pred.shape[1],
            device=y_pred.device,
            dtype=y_pred.dtype,
        )

        for i in range(0, y_pred.shape[1]):
            if mask is not None:
                # diff[:, i] has shape (N, T, H, W) or (N, H, W) depending on sequence logic
                # Calculate elements per patch for this specific variable
                elements_per_var_per_patch = diff[:, i].numel() // diff.shape[0]

                # Mean over only the active patches
                batch_mean[i] = diff[:, i].sum() / (mask.sum() * elements_per_var_per_patch + 1e-8)
            else:
                batch_mean[i] = diff[:, i].mean()

            if batch_mean[i] < 0:
                print(f"Negative mean for var {i}: {batch_mean[i]}")

        return batch_mean
