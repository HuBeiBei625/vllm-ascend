from dataclasses import dataclass
from typing import Any

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.utils import extract_layer_index

from vllm_ascend.ascend_config import get_ascend_config


@dataclass(frozen=True)
class Qwen36PrefillTTFTDecision:
    enabled: bool
    reason: str
    is_prefill: bool = False


def is_qwen36_model_path(vllm_config: VllmConfig) -> bool:
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        return False

    candidates: list[str] = []
    for config_name in ("hf_text_config", "hf_config"):
        hf_config = getattr(model_config, config_name, None)
        if hf_config is None:
            continue
        model_type = getattr(hf_config, "model_type", None)
        if model_type is not None:
            candidates.append(str(model_type))
        architectures = getattr(hf_config, "architectures", None)
        if architectures is not None:
            candidates.extend(str(arch) for arch in architectures)

    return any(
        "qwen3_5" in candidate.lower()
        or "qwen3.5" in candidate.lower()
        or "qwen3_6" in candidate.lower()
        or "qwen3.6" in candidate.lower()
        for candidate in candidates
    )


def _pick_layer_metadata(attn_metadata: Any, layer_prefix: str | None) -> Any:
    if not isinstance(attn_metadata, dict):
        return attn_metadata

    if layer_prefix in attn_metadata:
        return attn_metadata[layer_prefix]

    if layer_prefix is not None:
        for key, value in attn_metadata.items():
            if layer_prefix in str(key) or str(key) in layer_prefix:
                return value

        try:
            layer_idx = extract_layer_index(layer_prefix)
        except (AssertionError, ValueError):
            layer_idx = None
        if layer_idx is not None:
            needle = f".layers.{layer_idx}."
            for key, value in attn_metadata.items():
                if needle in str(key):
                    return value

    if len(attn_metadata) == 1:
        return next(iter(attn_metadata.values()))

    return None


def _is_pure_prefill_metadata(metadata: Any) -> bool:
    if metadata is None:
        return False

    spec_sequence_masks = getattr(metadata, "spec_sequence_masks", None)
    if spec_sequence_masks is not None:
        return False

    num_decodes = getattr(metadata, "num_decodes", 0) or 0
    num_decode_tokens = getattr(metadata, "num_decode_tokens", 0) or 0
    if num_decodes > 0 or num_decode_tokens > 0:
        return False

    num_prefills = getattr(metadata, "num_prefills", None)
    if num_prefills is not None:
        return num_prefills > 0

    attn_state = getattr(metadata, "attn_state", None)
    if attn_state is not None:
        return getattr(attn_state, "name", "") in {
            "PrefillNoCache",
            "PrefillCacheHit",
            "ChunkedPrefill",
        }

    return False


def get_qwen36_prefill_ttft_decision(
    vllm_config: VllmConfig,
    layer_prefix: str | None = None,
    attn_metadata: Any | None = None,
) -> Qwen36PrefillTTFTDecision:
    try:
        opt_config = get_ascend_config().qwen36_prefill_ttft_opt
    except (AttributeError, RuntimeError):
        return Qwen36PrefillTTFTDecision(False, "ascend config is not initialized")

    if not opt_config.enabled:
        return Qwen36PrefillTTFTDecision(False, "qwen36_prefill_ttft_opt is disabled")

    if not is_qwen36_model_path(vllm_config):
        return Qwen36PrefillTTFTDecision(False, "model is not Qwen3.6/Qwen3.5 path")

    parallel_config = vllm_config.parallel_config
    if parallel_config.prefill_context_parallel_size <= 1:
        return Qwen36PrefillTTFTDecision(False, "prefill context parallel is not enabled")

    if attn_metadata is None:
        try:
            attn_metadata = get_forward_context().attn_metadata
        except AssertionError:
            attn_metadata = None

    layer_metadata = _pick_layer_metadata(attn_metadata, layer_prefix)
    is_prefill = _is_pure_prefill_metadata(layer_metadata)
    if not is_prefill:
        return Qwen36PrefillTTFTDecision(False, "batch is not pure prefill", is_prefill=False)

    return Qwen36PrefillTTFTDecision(True, "prefill eligible", is_prefill=True)


def is_qwen36_prefill_ttft_opt_enabled(
    vllm_config: VllmConfig,
    layer_prefix: str | None = None,
    attn_metadata: Any | None = None,
) -> bool:
    return get_qwen36_prefill_ttft_decision(vllm_config, layer_prefix, attn_metadata).enabled
