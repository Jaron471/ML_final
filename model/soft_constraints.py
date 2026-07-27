"""Differentiable PDE constraints that regularize, but never replace, a CNN."""

from __future__ import annotations

import torch
import torch.nn as nn


class SoftPDEConstraintLoss(nn.Module):
    """Training-only u-field identities for the three inverse PDE problems.

    This module never solves for or overwrites parameters.  Gradients flow
    only into the CNN parameter predictions; the observed ``u`` field
    supplies fixed coefficients for the physical identities.
    """

    def __init__(
        self,
        pde: str,
        grid_size: int = 128,
        dx: float = 1.0,
    ) -> None:
        super().__init__()
        if pde not in (
            "schnakenberg",
            "brusselator",
            "lengyel_epstein",
        ):
            raise ValueError(f"Unsupported PDE: {pde}")
        self.pde = pde
        self.grid_size = int(grid_size)
        frequencies = (
            2.0
            * torch.pi
            * torch.fft.fftfreq(
                self.grid_size,
                d=float(dx),
                dtype=torch.float64,
            )
        )
        laplace_eigenvalue = -(
            frequencies[:, None] ** 2
            + frequencies[None, :] ** 2
        )
        self.register_buffer(
            "laplace_eigenvalue",
            laplace_eigenvalue,
        )

    def _laplacian(self, value: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifft2(
            self.laplace_eigenvalue * torch.fft.fft2(value)
        ).real

    @staticmethod
    def _relative_field_loss(
        residual: torch.Tensor,
        terms: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Mean per-sample residual energy normalized by equation scale."""
        numerator = residual.square().mean(dim=(1, 2))
        denominator = sum(
            term.detach().square().mean(dim=(1, 2))
            for term in terms
        ).clamp_min(1e-16)
        return (numerator / denominator).mean()

    def forward(
        self,
        u: torch.Tensor,
        params_real: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if u.shape[1:] != (
            1,
            self.grid_size,
            self.grid_size,
        ):
            raise ValueError(
                "Expected u with shape "
                f"(batch,1,{self.grid_size},{self.grid_size})."
            )
        if params_real.shape[1] != 3:
            raise ValueError("Expected three predicted parameters.")

        prediction_dtype = params_real.dtype
        params = params_real.to(torch.float64)
        with torch.no_grad():
            field = u[:, 0].to(torch.float64)
            mean_u = field.mean(dim=(1, 2))

        if self.pde == "schnakenberg":
            a, b = params[:, 0], params[:, 1]
            identity = (a + b - mean_u).square().mean()
            elimination = identity.new_zeros(())
            return (
                identity.to(prediction_dtype),
                elimination.to(prediction_dtype),
            )

        with torch.no_grad():
            lap_u = self._laplacian(field)

        if self.pde == "brusselator":
            A = params[:, 0, None, None]
            B = params[:, 1, None, None]
            d = params[:, 2, None, None]
            with torch.no_grad():
                reciprocal_u = field.reciprocal()
                column_dB = self._laplacian(reciprocal_u)
                column_d = self._laplacian(
                    (field - lap_u) / field.square()
                )
                column_dA = self._laplacian(
                    field.reciprocal().square()
                )
                constant = lap_u - field
            term_dB = d * B * column_dB
            term_d = d * column_d
            term_dA = -d * A * column_dA
            term_constant = constant + A
            residual = term_dB + term_d + term_dA + term_constant
            elimination = self._relative_field_loss(
                residual,
                (term_dB, term_d, term_dA, term_constant),
            )
            identity = (
                params[:, 0] - mean_u
            ).square().mean()
            return (
                identity.to(prediction_dtype),
                elimination.to(prediction_dtype),
            )

        a = params[:, 0, None, None]
        phi = params[:, 1, None, None]
        r = params[:, 2, None, None]
        c = a - phi
        with torch.no_grad():
            inverse_q = (1.0 + field.square()) / field
            column_rc = self._laplacian(inverse_q)
            column_r = self._laplacian(
                (lap_u - field) * inverse_q
            )
            constant = (
                5.0 * field
                - lap_u
                - 5.0 * mean_u[:, None, None]
            )
        term_rc = r * c * column_rc
        term_r = r * column_r
        residual = term_rc + term_r + constant
        elimination = self._relative_field_loss(
            residual,
            (term_rc, term_r, constant),
        )
        # Divide by 25 so the integrated identity is on an order-one scale.
        identity = (
            (
                params[:, 0]
                - 5.0 * params[:, 1]
                - 5.0 * mean_u
            ).square().mean()
            / 25.0
        )
        return (
            identity.to(prediction_dtype),
            elimination.to(prediction_dtype),
        )


class GMSoftConstraintLoss(nn.Module):
    """Training-only global constraints for Gierer--Meinhardt.

    For the steady periodic system

        0 = s Δu + a - bu + u² / (v(1 + cu²))
        0 = s δ Δv + u² - v,

    integrating the first equation removes the Laplacian.  Multiplying the
    second equation by Δv and integrating gives a scalar normal equation for
    ``delta``.  Both are used only as differentiable penalties; this module
    never solves for or overwrites a predicted parameter.
    """

    def __init__(
        self,
        grid_size: int = 128,
        dx: float = 1.0,
        s_diffusion: float = 0.4,
    ) -> None:
        super().__init__()
        self.grid_size = int(grid_size)
        self.s_diffusion = float(s_diffusion)
        frequencies = (
            2.0
            * torch.pi
            * torch.fft.fftfreq(
                self.grid_size,
                d=float(dx),
                dtype=torch.float64,
            )
        )
        laplace_eigenvalue = -(
            frequencies[:, None] ** 2
            + frequencies[None, :] ** 2
        )
        self.register_buffer(
            "laplace_eigenvalue",
            laplace_eigenvalue,
        )

    def _laplacian(self, value: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifft2(
            self.laplace_eigenvalue * torch.fft.fft2(value)
        ).real

    @staticmethod
    def _relative_scalar_loss(
        residual: torch.Tensor,
        terms: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        denominator = sum(
            term.detach().square() for term in terms
        ).clamp_min(1e-16)
        return (residual.square() / denominator).mean()

    def forward(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        params_real: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (
            1,
            self.grid_size,
            self.grid_size,
        )
        if u.shape[1:] != expected or v.shape[1:] != expected:
            raise ValueError(
                "Expected u and v with shape "
                f"(batch,1,{self.grid_size},{self.grid_size})."
            )
        if params_real.shape[1] != 4:
            raise ValueError(
                "GMSoftConstraintLoss expects [a, b, c, delta]."
            )

        prediction_dtype = params_real.dtype
        params = params_real.to(torch.float64)
        a, b, c, delta = params.unbind(dim=1)
        with torch.no_grad():
            field_u = u[:, 0].to(torch.float64)
            field_v = v[:, 0].to(torch.float64).clamp_min(1e-6)
            mean_u = field_u.mean(dim=(1, 2))
            lap_v = self._laplacian(field_v)
            lap_v_energy = lap_v.square().mean(dim=(1, 2))
            delta_constant = (
                lap_v * (field_u.square() - field_v)
            ).mean(dim=(1, 2))

        reaction = (
            field_u.square()
            / (
                field_v
                * (
                    1.0
                    + c[:, None, None] * field_u.square()
                )
            )
        )
        term_a = a
        term_b = -b * mean_u
        term_reaction = reaction.mean(dim=(1, 2))
        integrated_residual = term_a + term_b + term_reaction
        integrated_loss = self._relative_scalar_loss(
            integrated_residual,
            (term_a, term_b, term_reaction),
        )

        term_delta = (
            self.s_diffusion * delta * lap_v_energy
        )
        delta_residual = term_delta + delta_constant
        delta_loss = self._relative_scalar_loss(
            delta_residual,
            (term_delta, delta_constant),
        )
        return (
            integrated_loss.to(prediction_dtype),
            delta_loss.to(prediction_dtype),
        )
