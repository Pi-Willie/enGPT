import torch

from engpt.kernels import carried_residual, reference_residual


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
