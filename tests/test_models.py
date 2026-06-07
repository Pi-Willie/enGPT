import torch

from engpt import EfficientNGPT, GPTBaseline, ModelConfig, build_gpt_adamw, build_ngpt_adamw
from engpt.optim import project_ngpt_gradients_, project_ngpt_parameters_


def small_cfg():
    return ModelConfig(
        vocab_size=32,
        block_size=12,
        n_layer=2,
        n_head=2,
        n_embd=32,
        mlp_ratio=2.0,
        dropout=0.0,
        alpha_init=0.1,
    )


def test_engpt_forward_matches_materialized_ngpt_reference():
    torch.manual_seed(1)
    cfg = small_cfg()
    model = EfficientNGPT(cfg).double()
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    logits, loss = model(idx, targets)
    ref_logits, ref_loss = model.forward_reference(idx, targets)
    # The carried residual uses the algebraic one-reduction form, so floating
    # point reductions are not bit-identical to the materialized reference.
    assert torch.allclose(logits, ref_logits, atol=1e-7, rtol=1e-6)
    assert torch.allclose(loss, ref_loss, atol=1e-8, rtol=1e-7)


def test_gpt_baseline_and_engpt_train_step():
    torch.manual_seed(2)
    cfg = small_cfg()
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    targets = (idx + 1) % cfg.vocab_size

    gpt = GPTBaseline(cfg)
    gpt_opt = build_gpt_adamw(gpt, lr=1e-3)
    _, gpt_loss = gpt(idx, targets)
    gpt_loss.backward()
    gpt_opt.step()

    engpt = EfficientNGPT(cfg)
    engpt_opt = build_ngpt_adamw(engpt, lr=1e-3)
    _, engpt_loss = engpt(idx, targets)
    engpt_loss.backward()
    engpt_opt.step()
    project_ngpt_parameters_(engpt)
    report = engpt.parameter_norm_report()
    assert report["max_unit_norm_error"] < 2e-6


def test_ngpt_gradient_projection_is_tangent():
    torch.manual_seed(3)
    cfg = small_cfg()
    model = EfficientNGPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    _, loss = model(idx, (idx + 1) % cfg.vocab_size)
    loss.backward()
    project_ngpt_gradients_(model)

    q = model.blocks[0].qkv.weight[: cfg.n_embd]
    qg = model.blocks[0].qkv.weight.grad[: cfg.n_embd]
    assert torch.allclose((q * qg).sum(dim=-1), torch.zeros(cfg.n_embd), atol=1e-5)

    out = model.blocks[0].out_proj.weight
    outg = model.blocks[0].out_proj.weight.grad
    assert torch.allclose((out * outg).sum(dim=0), torch.zeros(cfg.n_embd), atol=1e-5)


def test_tst_bag_size_one_matches_normal_forward():
    torch.manual_seed(4)
    cfg = small_cfg()
    model = EfficientNGPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    logits, loss = model(idx, targets)
    tst_logits, tst_loss = model.forward_tst_superposition(idx, targets, bag_size=1)
    assert torch.allclose(tst_logits, logits, atol=1e-6, rtol=1e-5)
    assert torch.allclose(tst_loss, loss, atol=1e-6, rtol=1e-5)


def test_tst_carried_bags_represent_unit_centroids():
    torch.manual_seed(5)
    cfg = small_cfg()
    model = EfficientNGPT(cfg)
    bag_size = 3
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    y, rho = model.carried_embedding_bags(idx, bag_size=bag_size)
    h = y / rho.unsqueeze(-1)
    assert y.shape == (2, cfg.block_size // bag_size, cfg.n_embd)
    assert rho.shape == y.shape[:2]
    assert torch.allclose(torch.linalg.vector_norm(h, dim=-1), torch.ones_like(rho), atol=1e-6)


def test_tst_multitarget_loss_is_mean_ce_over_next_bag():
    torch.manual_seed(6)
    cfg = small_cfg()
    model = EfficientNGPT(cfg)
    bag_size = 2
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    logits, loss = model.forward_tst_superposition(idx, targets, bag_size=bag_size)
    target_bags = targets.view(targets.shape[0], targets.shape[1] // bag_size, bag_size)
    expected = -torch.nn.functional.log_softmax(logits.float(), dim=-1).gather(-1, target_bags).mean()
    assert torch.allclose(loss, expected, atol=1e-7, rtol=1e-7)
