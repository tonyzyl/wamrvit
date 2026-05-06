# -----------------------------------------------------------------------------
# Attribution:
# This code is adapted from the benchmark metrics of "The Well" dataset
# developed by Polymathic AI.
# Original repository: https://github.com/PolymathicAI/the_well
# -----------------------------------------------------------------------------

import torch


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
    "VMSE": VMSE,
    "LInfinity": LInfinity,
    "PearsonR": PearsonR,
}
