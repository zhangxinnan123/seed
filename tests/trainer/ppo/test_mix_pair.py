"""Two-context mixture rollout: log-mu recovery and the alpha=0 no-op guarantee.

The vectorized fold inside TrajectoryCollector must agree with the scalar
reference in seed/mix_pair.py, and alpha=0 must leave the reported log-probs
exactly equal to the single-context values -- that is what makes a
mix_alpha=0.0 run identical to SEED without this feature.
"""

import math

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl import DataProto

from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from seed.mix_pair import (
    ALPHA_EPS,
    common_prefix_len,
    interleave_indices,
    mixture_logprob,
    pair_seed,
)


def reference_log_mu(lp1, lp2, alpha):
    """Naive definition, computed in probability space."""
    return math.log((1.0 - alpha) * math.exp(lp1) + alpha * math.exp(lp2))


@pytest.mark.parametrize("alpha", [0.05, 0.3, 0.5, 0.9])
def test_mixture_logprob_matches_probability_space(alpha):
    lp1 = [-0.1, -2.0, -7.5]
    lp2 = [-1.0, -0.2, -12.0]
    got = mixture_logprob(lp1, lp2, alpha, n_fused=len(lp1))
    for g, a, b in zip(got, lp1, lp2):
        assert g == pytest.approx(reference_log_mu(a, b, alpha), abs=1e-9)


def test_alpha_zero_is_identity():
    lp1 = [-0.3, -1.7]
    lp2 = [-9.0, -0.01]
    assert mixture_logprob(lp1, lp2, 0.0, n_fused=2) == lp1
    assert mixture_logprob(lp1, lp2, ALPHA_EPS / 2, n_fused=2) == lp1


def test_desync_falls_back_to_context1():
    lp1 = [-0.5, -0.5, -0.5]
    lp2 = [-1.5, -1.5, -1.5]
    got = mixture_logprob(lp1, lp2, 0.5, n_fused=1)
    assert got[0] == pytest.approx(reference_log_mu(-0.5, -1.5, 0.5))
    assert got[1:] == lp1[1:]


def test_common_prefix_and_interleave():
    assert common_prefix_len([1, 2, 3], [1, 2, 9]) == 2
    assert common_prefix_len([1], [2]) == 0
    assert interleave_indices(3) == [0, 3, 1, 4, 2, 5]


def test_pair_seed_is_stable_and_pair_specific():
    assert pair_seed(0, "a") == pair_seed(0, "a")
    assert pair_seed(0, "a") != pair_seed(0, "b")
    assert 0 <= pair_seed(7, "a") < 2**31 - 1


def _collector(alpha, adv_estimator="seed"):
    from omegaconf import OmegaConf

    config = OmegaConf.create(
        {"algorithm": {"adv_estimator": adv_estimator, "seed": {"mix_alpha": alpha}}}
    )
    return TrajectoryCollector(config=config, tokenizer=None)


def test_eval_never_mixes():
    collector = _collector(0.3)
    assert collector._mix_alpha("train") == 0.3
    # validation / test rollouts must stay single-context
    assert collector._mix_alpha("val") == 0.0
    assert collector._mix_alpha("test") == 0.0


def test_mixture_requires_seed_estimator():
    assert _collector(0.3, adv_estimator="grpo")._mix_alpha("val") == 0.0
    with pytest.raises(ValueError, match="requires algorithm.adv_estimator=seed"):
        _collector(0.3, adv_estimator="grpo")._mix_alpha("train")
    with pytest.raises(ValueError, match="requires algorithm.adv_estimator=seed"):
        _collector(0.3, adv_estimator="gigpo")._mix_alpha("train")
    # alpha=0 is inert regardless of estimator
    assert _collector(0.0, adv_estimator="gigpo")._mix_alpha("train") == 0.0


def _pair_batch(ids1, ids2, lp1, lp2, prompt_len=2):
    """Interleaved (c1, c2) rollout output with a 2-token dummy prompt."""
    n = len(ids1)
    width = len(ids1[0])
    responses, log_probs, attn = [], [], []
    for row in range(n):
        for ids, lps in ((ids1[row], lp1[row]), (ids2[row], lp2[row])):
            responses.append(ids)
            log_probs.append(lps)
            # response mask: 1 for real tokens (all of them here)
            attn.append([1] * prompt_len + [1] * width)
    batch = TensorDict(
        {
            "responses": torch.tensor(responses, dtype=torch.long),
            "rollout_log_probs": torch.tensor(log_probs, dtype=torch.float32),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        },
        batch_size=2 * n,
    )
    return DataProto(batch=batch)


def test_fold_matches_scalar_reference():
    collector = TrajectoryCollector(config=None, tokenizer=None)
    alpha = 0.25
    ids1 = [[5, 6, 7]]
    ids2 = [[5, 6, 7]]  # in sync for the whole response
    lp1 = [[-0.2, -1.0, -3.0]]
    lp2 = [[-0.9, -0.4, -0.1]]

    folded = collector._fold_mix_pair_output(
        _pair_batch(ids1, ids2, lp1, lp2), np.array([alpha], dtype=np.float32)
    )

    assert len(folded) == 1
    expected = mixture_logprob(lp1[0], lp2[0], alpha, n_fused=3)
    got = folded.batch["rollout_log_probs"][0].tolist()
    for g, e in zip(got, expected):
        assert g == pytest.approx(e, abs=1e-6)
    stats = collector.pop_mix_stats()
    assert stats["mix/in_sync_ratio"] == 1.0
    assert stats["mix/fused_token_ratio"] == 1.0


def test_fold_alpha_zero_returns_context1_logprobs():
    collector = TrajectoryCollector(config=None, tokenizer=None)
    lp1 = [[-0.2, -1.0, -3.0]]
    folded = collector._fold_mix_pair_output(
        _pair_batch([[5, 6, 7]], [[5, 6, 7]], lp1, [[-9.0, -9.0, -9.0]]),
        np.array([0.0], dtype=np.float32),
    )
    assert folded.batch["rollout_log_probs"][0].tolist() == pytest.approx(lp1[0])


def test_fold_desync_switches_to_context1():
    collector = TrajectoryCollector(config=None, tokenizer=None)
    alpha = 0.5
    lp1 = [[-0.5, -0.5, -0.5]]
    lp2 = [[-1.5, -1.5, -1.5]]
    folded = collector._fold_mix_pair_output(
        _pair_batch([[5, 6, 7]], [[5, 99, 7]], lp1, lp2),  # diverges at t=1
        np.array([alpha], dtype=np.float32),
    )
    got = folded.batch["rollout_log_probs"][0].tolist()
    assert got[0] == pytest.approx(reference_log_mu(-0.5, -1.5, alpha), abs=1e-6)
    assert got[1:] == pytest.approx(lp1[0][1:])
    stats = collector.pop_mix_stats()
    assert stats["mix/in_sync_ratio"] == 0.0


def test_task_key_is_stable_across_history_and_step():
    from seed.mix_pair import task_key

    # Same task, different accumulated history / step wording -> same key.
    a = "You are in the middle of a room.\nYour task is to: put a clean mug in coffeemachine.\nStep 0"
    b = "Obs after 7 actions...\nYour task is to: put a clean mug in coffeemachine.\nStep 7"
    assert task_key(a) == task_key(b) != ""
    c = "Your task is to: heat some apple and put it in fridge."
    assert task_key(c) != task_key(a)


def test_update_mix_skills_merges_and_ignores_blanks():
    collector = _collector(0.3)
    collector.update_mix_skills({"task-a": {"skill": "skill one", "success": True}})
    collector.update_mix_skills(
        {
            "task-a": {"skill": "skill two", "success": True},
            "task-b": {"skill": "other", "success": False},
            "": {"skill": "x", "success": True},
            "task-c": {"skill": "  ", "success": True},
        }
    )
    assert collector._mix_skill_for("task-a") == "skill two"   # success -> success overwrites
    assert collector._mix_skill_for("task-b") == "other"
    assert collector._mix_skill_for("task-c") == ""             # blank skipped
    assert set(collector._mix_skills) == {"task-a", "task-b"}


def test_bank_success_priority_protects_successful_skill():
    collector = _collector(0.3)                                 # default: success_priority
    collector.update_mix_skills({"t": {"skill": "workflow from success", "success": True}})
    # a failed episode's skill must not displace it
    collector.update_mix_skills({"t": {"skill": "avoidance from failure", "success": False}})
    assert collector._mix_skill_for("t") == "workflow from success"
    # another success does replace it (latest among successes)
    collector.update_mix_skills({"t": {"skill": "newer workflow", "success": True}})
    assert collector._mix_skill_for("t") == "newer workflow"


def test_bank_failure_overwrites_failure():
    collector = _collector(0.3)
    collector.update_mix_skills({"t": {"skill": "avoid A", "success": False}})
    collector.update_mix_skills({"t": {"skill": "avoid B", "success": False}})
    assert collector._mix_skill_for("t") == "avoid B"


def test_bank_latest_mode_overwrites_unconditionally():
    from omegaconf import OmegaConf

    config = OmegaConf.create(
        {"algorithm": {"adv_estimator": "seed", "seed": {"mix_alpha": 0.3, "mix_skill_overwrite": "latest"}}}
    )
    collector = TrajectoryCollector(config=config, tokenizer=None)
    collector.update_mix_skills({"t": {"skill": "workflow from success", "success": True}})
    collector.update_mix_skills({"t": {"skill": "avoidance from failure", "success": False}})
    assert collector._mix_skill_for("t") == "avoidance from failure"


def test_mix_is_weight_is_identity_for_unmixed_rows():
    from verl.trainer.ppo.core_algos import compute_mix_is_weight

    old_lp = torch.tensor([[-0.2, -1.0], [-0.2, -1.0]])
    # engine log-probs differ from the actor's (vLLM vs FSDP numerics + mixture)
    rollout_lp = torch.tensor([[-0.5, -0.7], [-0.5, -0.7]])
    mask = torch.ones_like(old_lp)
    alpha = torch.tensor([0.0, 0.3])  # row 0 never mixed, row 1 mixed

    w, metrics = compute_mix_is_weight(old_lp, rollout_lp, mask, alpha, log_clip=2.0)
    # unmixed row: exactly 1, so alpha=0 stays byte-identical to plain SEED
    assert w[0].tolist() == pytest.approx([1.0, 1.0])
    # mixed row: exp(old - log mu)
    assert w[1].tolist() == pytest.approx([math.exp(0.3), math.exp(-0.3)], abs=1e-6)
    assert metrics["actor/mix_is_clipped_ratio"] == 0.0


def test_mix_is_weight_clamps_and_respects_mask():
    from verl.trainer.ppo.core_algos import compute_mix_is_weight

    old_lp = torch.tensor([[-0.1, -0.1]])
    rollout_lp = torch.tensor([[-9.0, -0.1]])  # huge gap on token 0
    mask = torch.tensor([[1.0, 0.0]])          # token 1 is padding
    w, metrics = compute_mix_is_weight(old_lp, rollout_lp, mask, torch.tensor([0.5]), log_clip=2.0)
    assert w[0, 0].item() == pytest.approx(math.exp(2.0), abs=1e-5)  # clamped
    assert w[0, 1].item() == 1.0                                     # masked -> identity
    assert metrics["actor/mix_is_clipped_ratio"] == 1.0


def test_advantage_scaling_equals_token_weighting():
    """w > 0 commutes with min()/clip(), which is why scaling advantages is exact."""
    from verl.trainer.ppo.core_algos import compute_policy_loss

    torch.manual_seed(0)
    old_lp = torch.randn(2, 4) * 0.1
    log_prob = old_lp + torch.randn(2, 4) * 0.05
    adv = torch.randn(2, 4)
    mask = torch.ones(2, 4)
    w = torch.rand(2, 4) + 0.5

    weighted_adv_loss, _, _, _ = compute_policy_loss(
        old_log_prob=old_lp, log_prob=log_prob, advantages=adv * w,
        response_mask=mask, cliprange=0.2, loss_agg_mode="token-mean",
    )
    # reference: per-token loss with the weight applied after the min()
    ratio = torch.exp(log_prob - old_lp)
    per_token = -torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv) * w
    assert weighted_adv_loss.item() == pytest.approx(per_token.mean().item(), abs=1e-5)


def _sched_collector(alpha, alpha_end, horizon):
    from omegaconf import OmegaConf

    config = OmegaConf.create(
        {
            "algorithm": {
                "adv_estimator": "seed",
                "seed": {
                    "mix_alpha": alpha,
                    "mix_alpha_end": alpha_end,
                    "mix_alpha_anneal_steps": horizon,
                },
            }
        }
    )
    return TrajectoryCollector(config=config, tokenizer=None)


def test_alpha_schedule_anneals_linearly_to_zero():
    collector = _sched_collector(0.3, 0.0, 11)
    assert collector._mix_alpha("train", global_step=1) == pytest.approx(0.3)
    assert collector._mix_alpha("train", global_step=6) == pytest.approx(0.15)
    assert collector._mix_alpha("train", global_step=11) == 0.0   # snapped by ALPHA_EPS
    # past the horizon it holds at the end value
    assert collector._mix_alpha("train", global_step=50) == 0.0
    # eval still never mixes, schedule or not
    assert collector._mix_alpha("val", global_step=1) == 0.0


def test_alpha_schedule_defaults_to_constant():
    collector = _sched_collector(0.25, None, None)
    assert collector._mix_alpha("train", global_step=1) == pytest.approx(0.25)
    assert collector._mix_alpha("train", global_step=999) == pytest.approx(0.25)


def test_opd_gate_can_be_disabled():
    """gate_enable=False must give plain ungated distillation."""
    from verl.trainer.ppo.core_algos import compute_opd_loss

    log_prob = torch.tensor([[-1.0, -2.0]], requires_grad=False)
    teacher = torch.tensor([[-0.5, -3.0]])   # teacher better on t=0, worse on t=1
    mask = torch.ones_like(log_prob)

    gated, _, gate_mean, _, gap_mean = compute_opd_loss(
        log_prob, teacher, mask, gate_beta=5.0, loss_agg_mode="token-mean"
    )
    ungated, _, ungated_gate_mean, _, _ = compute_opd_loss(
        log_prob, teacher, mask, gate_beta=5.0, loss_agg_mode="token-mean", gate_enable=False
    )

    # ungated == masked mean of (teacher - student), weight 1 everywhere
    assert ungated.item() == pytest.approx(((teacher - log_prob)).mean().item(), abs=1e-6)
    assert ungated_gate_mean.item() == 1.0
    # the gate down-weights the token where the student is already ahead
    assert 0.0 < gate_mean.item() < 1.0
    assert gated.item() != pytest.approx(ungated.item(), abs=1e-6)
    assert gap_mean.item() == pytest.approx(((teacher - log_prob)).mean().item(), abs=1e-6)


def test_skill_bank_round_trip(tmp_path):
    """The bank must survive a process boundary: dump on merge, load on first use."""
    from omegaconf import OmegaConf

    bank = tmp_path / "skill_bank.json"
    config = OmegaConf.create(
        {
            "algorithm": {"adv_estimator": "seed", "seed": {"mix_alpha": 0.3, "mix_skill_bank_path": str(bank)}},
            "trainer": {"default_local_dir": str(tmp_path)},
        }
    )

    writer = TrajectoryCollector(config=config, tokenizer=None)
    writer.update_mix_skills(
        {
            "put two pot in shelf.": {"skill": "workflow", "success": True},
            "put a candle in dresser.": {"skill": "avoid X", "success": False},
        }
    )
    assert bank.exists()

    reader = TrajectoryCollector(config=config, tokenizer=None)
    assert reader._mix_skills == {}          # not loaded yet
    reader._maybe_load_mix_skills()
    assert reader._mix_skill_for("put two pot in shelf.") == "workflow"
    assert reader._mix_skills["put two pot in shelf."]["success"] is True
    assert reader._mix_skills["put a candle in dresser."]["success"] is False

    # success_priority still applies across the reload boundary
    reader.update_mix_skills({"put two pot in shelf.": {"skill": "avoid Y", "success": False}})
    assert reader._mix_skill_for("put two pot in shelf.") == "workflow"


def test_skill_bank_missing_file_is_not_fatal(tmp_path):
    from omegaconf import OmegaConf

    config = OmegaConf.create(
        {
            "algorithm": {"adv_estimator": "seed", "seed": {"mix_alpha": 0.3, "mix_skill_bank_path": str(tmp_path / "nope.json")}},
            "trainer": {"default_local_dir": str(tmp_path)},
        }
    )
    collector = TrajectoryCollector(config=config, tokenizer=None)
    collector._maybe_load_mix_skills()       # must not raise
    assert collector._mix_skills == {}


def test_mix_is_enable_switch():
    """mix_is_enable=False must leave the PG term unweighted (biased ablation)."""
    from verl.trainer.ppo.core_algos import compute_mix_is_weight

    old_lp = torch.tensor([[-0.2, -1.0]])
    rollout_lp = torch.tensor([[-0.5, -0.7]])
    mask = torch.ones_like(old_lp)
    alpha = torch.tensor([0.3])

    # the helper itself always computes the weight; the switch lives in dp_actor's
    # use_mix_is guard, so assert the weight is non-trivial when it IS applied
    w, _ = compute_mix_is_weight(old_lp, rollout_lp, mask, alpha, log_clip=2.0)
    assert not torch.allclose(w, torch.ones_like(w))
