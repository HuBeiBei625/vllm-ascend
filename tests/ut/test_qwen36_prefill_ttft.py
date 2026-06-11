from types import SimpleNamespace
from unittest.mock import patch

from vllm_ascend.qwen36_prefill_ttft import get_qwen36_prefill_ttft_decision


def _vllm_config(enabled=True, pcp_size=2, model_type="qwen3_5_moe"):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp_size),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                model_type=model_type,
                architectures=[],
            ),
            hf_config=None,
        ),
    )


def _ascend_config(enabled=True):
    return SimpleNamespace(
        qwen36_prefill_ttft_opt=SimpleNamespace(
            enabled=enabled,
        )
    )


def test_qwen36_prefill_ttft_enabled_for_pure_prefill_metadata():
    metadata = SimpleNamespace(num_decodes=0, num_decode_tokens=0, num_prefills=1)

    with patch("vllm_ascend.qwen36_prefill_ttft.get_ascend_config", return_value=_ascend_config(enabled=True)):
        decision = get_qwen36_prefill_ttft_decision(
            _vllm_config(),
            layer_prefix="model.layers.0.self_attn",
            attn_metadata={"model.layers.0.self_attn": metadata},
        )

    assert decision.enabled
    assert decision.is_prefill


def test_qwen36_prefill_ttft_rejects_decode_metadata():
    metadata = SimpleNamespace(num_decodes=1, num_decode_tokens=1, num_prefills=0)

    with patch("vllm_ascend.qwen36_prefill_ttft.get_ascend_config", return_value=_ascend_config(enabled=True)):
        decision = get_qwen36_prefill_ttft_decision(
            _vllm_config(),
            layer_prefix="model.layers.0.self_attn",
            attn_metadata={"model.layers.0.self_attn": metadata},
        )

    assert not decision.enabled
    assert not decision.is_prefill
    assert decision.reason == "batch is not pure prefill"


def test_qwen36_prefill_ttft_rejects_non_qwen_model():
    metadata = SimpleNamespace(num_decodes=0, num_decode_tokens=0, num_prefills=1)

    with patch("vllm_ascend.qwen36_prefill_ttft.get_ascend_config", return_value=_ascend_config(enabled=True)):
        decision = get_qwen36_prefill_ttft_decision(
            _vllm_config(model_type="llama"),
            layer_prefix="model.layers.0.self_attn",
            attn_metadata={"model.layers.0.self_attn": metadata},
        )

    assert not decision.enabled
    assert decision.reason == "model is not Qwen3.6/Qwen3.5 path"


def test_qwen36_prefill_ttft_rejects_pcp_disabled():
    metadata = SimpleNamespace(num_decodes=0, num_decode_tokens=0, num_prefills=1)

    with patch("vllm_ascend.qwen36_prefill_ttft.get_ascend_config", return_value=_ascend_config(enabled=True)):
        decision = get_qwen36_prefill_ttft_decision(
            _vllm_config(pcp_size=1),
            layer_prefix="model.layers.0.self_attn",
            attn_metadata={"model.layers.0.self_attn": metadata},
        )

    assert not decision.enabled
    assert decision.reason == "prefill context parallel is not enabled"
