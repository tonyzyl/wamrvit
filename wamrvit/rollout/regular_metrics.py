# Benchmark metrics adapted from The Well, developed by Polymathic AI.
# Source: https://github.com/PolymathicAI/the_well/blob/v1.2.0/the_well/benchmark/metrics/spatial.py
# The following BSD-3-Clause notice applies to the upstream-derived portions.
#
# BSD 3-Clause License
#
# Copyright (c) 2024 Polymathic AI.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of Polymathic AI nor the names of the Well contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import torch


def normalize_numeric_result_columns(df):
    """Normalize scalar/singleton numeric Ray result cells to a float DataFrame.

    Supported sparse metrics can arrive as one-element ndarrays in object
    columns. Preserve None/NaN as missing support, and reject malformed cells
    instead of silently dropping columns or filling unsupported levels.
    """
    import pandas as pd

    numeric_columns = {}
    for column in df.columns:
        values = []
        for value in df[column]:
            if value is None:
                values.append(np.nan)
                continue
            array = np.asarray(value)
            if array.size != 1 or array.dtype.kind not in "biuf":
                raise TypeError(
                    f"Nonnumeric rollout metric value in {column}: "
                    f"shape={array.shape}, dtype={array.dtype}"
                )
            values.append(float(array.reshape(-1)[0]))
        numeric_columns[column] = values
    return pd.DataFrame(numeric_columns, index=df.index, dtype=float)


def mean_numeric_result_columns(df):
    """Average each result column over supported windows; all-missing stays NaN."""
    return normalize_numeric_result_columns(df).mean(skipna=True)


class PearsonR:
    @staticmethod
    def eval(
        x: torch.Tensor,
        y: torch.Tensor,
        n_spatial_dims: int,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        Pearson Correlation Coefficient

        Args:
            x: Input tensor.
            y: Target tensor.
            n_spatial_dims: Number of spatial dimensions in the data.
            eps: Small value to avoid division by zero.

        Returns:
            Pearson correlation coefficient between x and y.
        """
        x_flat = torch.flatten(x, start_dim=-n_spatial_dims - 1, end_dim=-2)
        y_flat = torch.flatten(y, start_dim=-n_spatial_dims - 1, end_dim=-2)

        # Calculate means along flattened axis
        x_mean = torch.mean(x_flat, dim=-2, keepdim=True)
        y_mean = torch.mean(y_flat, dim=-2, keepdim=True)

        # Calculate covariance
        covariance = torch.mean((x_flat - x_mean) * (y_flat - y_mean), dim=-2)
        # Calculate standard deviations
        std_x = torch.std(x_flat, dim=-2)
        std_y = torch.std(y_flat, dim=-2)

        # Calculate Pearson correlation coefficient
        correlation = covariance / (std_x * std_y + eps)
        return correlation


class MSE:
    @staticmethod
    def eval(
        x: torch.Tensor,
        y: torch.Tensor,
        n_spatial_dims: int,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        Mean Squared Error

        Args:
            x: Input tensor.
            y: Target tensor.
            n_spatial_dims: Number of spatial dimensions in the data.

        Returns:
            Mean squared error between x and y.
        """
        spatial_dims = tuple(range(-n_spatial_dims - 1, -1))
        return torch.mean((x - y) ** 2, dim=spatial_dims)


class MAE:
    @staticmethod
    def eval(
        x: torch.Tensor,
        y: torch.Tensor,
        n_spatial_dims: int,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        Mean Absolute Error

        Args:
            x: Input tensor.
            y: Target tensor.
            n_spatial_dims: Number of spatial dimensions in the data.

        Returns:
            Mean absolute error between x and y.
        """
        spatial_dims = tuple(range(-n_spatial_dims - 1, -1))
        return torch.mean((x - y).abs(), dim=spatial_dims)


class NMAE:
    @staticmethod
    def eval(
        x: torch.Tensor,
        y: torch.Tensor,
        n_spatial_dims: int,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        Normalized Mean Absolute Error

        Args:
            x: Input tensor.
            y: Target tensor.
            n_spatial_dims: Number of spatial dimensions in the data.

        Returns:
            Normalized mean absolute error between x and y.
        """
        spatial_dims = tuple(range(-n_spatial_dims - 1, -1))
        norm = torch.mean(torch.abs(y), dim=spatial_dims)
        return torch.mean((x - y).abs(), dim=spatial_dims) / (norm + eps)


class NMSE:
    @staticmethod
    def eval(
        x: torch.Tensor,
        y: torch.Tensor,
        n_spatial_dims: int,
        eps: float = 1e-7,
        norm_mode: str = "norm",
    ) -> torch.Tensor:
        """
        Normalized Mean Squared Error

        Args:
            x: Input tensor.
            y: Target tensor.
            n_spatial_dims: Number of spatial dimensions in the data.
            eps: Small value to avoid division by zero. Default is 1e-7.
            norm_mode: Mode for computing the normalization factor. Can be
                'norm' or 'std'. Default is 'norm'.

        Returns:
            Normalized mean squared error between x and y.
        """
        spatial_dims = tuple(range(-n_spatial_dims - 1, -1))
        if norm_mode == "norm":
            norm = torch.mean(y**2, dim=spatial_dims)
        elif norm_mode == "std":
            norm = torch.std(y, dim=spatial_dims) ** 2
        else:
            raise ValueError(f"Invalid norm_mode: {norm_mode}")
        return MSE.eval(x, y, n_spatial_dims) / (norm + eps)


class RMSE:
    @staticmethod
    def eval(
        x: torch.Tensor,
        y: torch.Tensor,
        n_spatial_dims: int,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        Root Mean Squared Error

        Args:
            x: Input tensor.
            y: Target tensor.
            n_spatial_dims: Number of spatial dimensions in the data.

        Returns:
            Root mean squared error between x and y.
        """
        return torch.sqrt(MSE.eval(x, y, n_spatial_dims))


class NRMSE:
    @staticmethod
    def eval(
        x: torch.Tensor,
        y: torch.Tensor,
        n_spatial_dims: int,
        eps: float = 1e-7,
        norm_mode: str = "norm",
    ) -> torch.Tensor:
        """
        Normalized Root Mean Squared Error

        Args:
            x: Input tensor.
            y: Target tensor.
            n_spatial_dims: Number of spatial dimensions in the data.
            eps: Small value to avoid division by zero. Default is 1e-7.
            norm_mode: Mode for computing the normalization factor. Can be
                'norm' or 'std'. Default is 'norm'.

        Returns:
            Normalized root mean squared error between x and y.
        """
        return torch.sqrt(NMSE.eval(x, y, n_spatial_dims, eps=eps, norm_mode=norm_mode))


class RelativeL2(NRMSE):
    """Per-channel relative L2 error, ``||x-y||_2 / ||y||_2``."""


class VMSE:
    @staticmethod
    def eval(
        x: torch.Tensor,
        y: torch.Tensor,
        n_spatial_dims: int,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        Variance Scaled Mean Squared Error

        Args:
            x: Input tensor.
            y: Target tensor.
            n_spatial_dims: Number of spatial dimensions in the data.

        Returns:
            Variance mean squared error between x and y.
        """
        return NMSE.eval(x, y, n_spatial_dims, norm_mode="std")


class VRMSE:
    @staticmethod
    def eval(
        x: torch.Tensor,
        y: torch.Tensor,
        n_spatial_dims: int,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        Root Variance Scaled Mean Squared Error

        Args:
            x: Input tensor.
            y: Target tensor.
            n_spatial_dims: Number of spatial dimensions in the data.

        Returns:
            Root variance mean squared error between x and y.
        """
        return NRMSE.eval(x, y, n_spatial_dims, norm_mode="std")


class LInfinity:
    @staticmethod
    def eval(
        x: torch.Tensor,
        y: torch.Tensor,
        n_spatial_dims: int,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        L-Infinity Norm

        Args:
            x: Input tensor.
            y: Target tensor.
            n_spatial_dims: Number of spatial dimensions in the data.

        Returns:
            L-Infinity norm between x and y.
        """
        spatial_dims = tuple(range(-n_spatial_dims - 1, -1))
        return torch.max(
            torch.abs(x - y).flatten(start_dim=spatial_dims[0], end_dim=-2), dim=-2
        ).values


METRIC_REGISTRY = {
    "RMSE": RMSE,
    "VRMSE": VRMSE,
    "MSE": MSE,
    "MAE": MAE,
    "NMAE": NMAE,
    "NMSE": NMSE,
    "NRMSE": NRMSE,
    "RelativeL2": RelativeL2,
    "VMSE": VMSE,
    "LInfinity": LInfinity,
    "PearsonR": PearsonR,
}


def eval_per_level_metric(
    metric_name: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Evaluate one per-channel metric over leaf tensors ``(N,H,W,C)``.

    Existing cell-space metrics retain the paper's mean-over-leaves protocol.
    ``RelativeL2`` instead reduces over all leaves and pixels so it equals the
    conventional per-channel norm ratio on the evaluated level.
    """
    metric_cls = METRIC_REGISTRY[metric_name]
    if metric_name == "RelativeL2":
        return metric_cls.eval(
            prediction.unsqueeze(0),
            target.unsqueeze(0),
            n_spatial_dims=3,
        )[0]
    return metric_cls.eval(prediction, target, n_spatial_dims=2).mean(dim=0)
