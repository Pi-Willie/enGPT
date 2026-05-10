import torch

from engpt.kernels import (
    carried_residual,
    carried_residual_gauge,
    gauge_carried_state,
    normalize_columns_,
    normalize_rows_,
    project_column_grad_,
    project_row_grad_,
    reference_residual,
    scaled_logits_from_carried,
)


def test_carried_residual_matches_reference_hidden_and_gradients():
    torch.manual_seed(0)
    y = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    rho = torch.rand(2, 3, dtype=torch.float64, requires_grad=True).add(0.5)
    rho.retain_grad()
    branch = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    alpha = torch.rand(8, dtype=torch.float64, requires_grad=True).mul(0.5)
    alpha.retain_grad()

    u, rho_plus = carried_residual(y, rho, branch, alpha, 1e-9, 1e-9)
    h_eff = u / rho_plus.unsqueeze(-1)
    h_ref = reference_residual(y / rho.unsqueeze(-1), branch, alpha, 1e-9, 1e-9)
    assert torch.allclose(h_eff, h_ref, atol=1e-10, rtol=1e-10)

    probe = torch.randn_like(h_ref)
    loss_eff = (h_eff * probe).sum()
    loss_eff.backward(retain_graph=True)
    grads_eff = (y.grad.clone(), rho.grad.clone(), branch.grad.clone(), alpha.grad.clone())

    y.grad = None
    rho.grad = None
    branch.grad = None
    alpha.grad = None
    loss_ref = (h_ref * probe).sum()
    loss_ref.backward()
    grads_ref = (y.grad, rho.grad, branch.grad, alpha.grad)
    for got, expected in zip(grads_eff, grads_ref):
        assert torch.allclose(got, expected, atol=1e-9, rtol=1e-8)


def test_carried_residual_gauge_matches_separate_residual_and_gauge():
    torch.manual_seed(4)
    y = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    rho = torch.rand(2, 3, dtype=torch.float64, requires_grad=True).add(0.5)
    rho.retain_grad()
    branch = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    alpha = torch.rand(8, dtype=torch.float64, requires_grad=True).mul(0.5)
    alpha.retain_grad()

    y_sep, rho_sep = carried_residual(y, rho, branch, alpha, 1e-9, 1e-9)
    y_sep, rho_sep = gauge_carried_state(y_sep, rho_sep, 0.75)
    y_fused, rho_fused = carried_residual_gauge(y, rho, branch, alpha, 1e-9, 1e-9, 0.75)
    assert torch.allclose(y_fused, y_sep, atol=1e-10, rtol=1e-10)
    assert torch.allclose(rho_fused, rho_sep, atol=1e-10, rtol=1e-10)

    probe_y = torch.randn_like(y_sep)
    probe_rho = torch.randn_like(rho_sep)
    loss_sep = (y_sep * probe_y).sum() + (rho_sep * probe_rho).sum()
    loss_sep.backward(retain_graph=True)
    grads_sep = (y.grad.clone(), rho.grad.clone(), branch.grad.clone(), alpha.grad.clone())

    y.grad = None
    rho.grad = None
    branch.grad = None
    alpha.grad = None
    loss_fused = (y_fused * probe_y).sum() + (rho_fused * probe_rho).sum()
    loss_fused.backward()
    grads_fused = (y.grad, rho.grad, branch.grad, alpha.grad)
    for got, expected in zip(grads_fused, grads_sep):
        assert torch.allclose(got, expected, atol=1e-9, rtol=1e-8)


def test_projection_helpers_match_torch_reference_on_available_devices():
    torch.manual_seed(5)
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for device in devices:
        weight = torch.randn(9, 13, device=device)

        rows = weight.clone()
        rows_ref = weight / torch.linalg.vector_norm(weight, dim=-1, keepdim=True).clamp_min(1e-6)
        normalize_rows_(rows, 1e-6)
        assert torch.allclose(rows, rows_ref, atol=1e-5, rtol=1e-5)

        cols = weight.clone()
        cols_ref = weight / torch.linalg.vector_norm(weight, dim=0, keepdim=True).clamp_min(1e-6)
        normalize_columns_(cols, 1e-6)
        assert torch.allclose(cols, cols_ref, atol=1e-5, rtol=1e-5)

        grad = torch.randn_like(weight)
        row_grad = grad.clone()
        row_grad_ref = grad - (grad * weight).sum(dim=-1, keepdim=True) * weight
        project_row_grad_(weight, row_grad)
        assert torch.allclose(row_grad, row_grad_ref, atol=1e-5, rtol=1e-5)

        col_grad = grad.clone()
        col_grad_ref = grad - weight * (grad * weight).sum(dim=0, keepdim=True)
        project_column_grad_(weight, col_grad)
        assert torch.allclose(col_grad, col_grad_ref, atol=1e-5, rtol=1e-5)


def test_scaled_logits_from_carried_matches_materialized_expression():
    torch.manual_seed(6)
    y = torch.randn(2, 3, 7, dtype=torch.float64)
    rho = torch.rand(2, 3, dtype=torch.float64).add(0.5)
    output_emb = torch.randn(11, 7, dtype=torch.float64)
    logit_scale = torch.rand(11, dtype=torch.float64).add(0.25)

    logits = scaled_logits_from_carried(y, rho, output_emb, logit_scale)
    expected = torch.nn.functional.linear(
        y / rho.unsqueeze(-1),
        output_emb * logit_scale.unsqueeze(-1),
    )
    assert torch.allclose(logits, expected, atol=1e-12, rtol=1e-12)
