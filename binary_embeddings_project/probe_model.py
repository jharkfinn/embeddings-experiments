"""Probe vLLM v0.18 model internals."""
import os
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"


def main():
    from vllm import LLM

    llm = LLM(
        model="/workspace/models/Qwen_Qwen3.5-35B-A3B-FP8",
        enforce_eager=True,
        gpu_memory_utilization=0.92,
        max_model_len=1056,
        trust_remote_code=True,
    )

    def probe(model):
        lines = []
        # model -> language_model -> model -> layers
        lm = model.language_model
        lines.append(f"language_model: {type(lm).__name__}")
        lm_attrs = [a for a in dir(lm) if not a.startswith("_")]
        lines.append(f"lm attrs: {lm_attrs[:20]}")
        if hasattr(lm, "model"):
            inner = lm.model
            lines.append(f"lm.model: {type(inner).__name__}")
            if hasattr(inner, "layers"):
                layers = list(inner.layers)
                lines.append(f"layers: {len(layers)}")
                l0 = layers[0]
                lines.append(f"layer0: {type(l0).__name__}")
                lines.append(f"layer0.layer_type: {getattr(l0, 'layer_type', 'N/A')}")
                l0a = [a for a in dir(l0) if not a.startswith("_")]
                lines.append(f"layer0 attrs: {l0a}")
                # layer 3 = full_attention
                l3 = layers[3]
                lines.append(f"layer3.layer_type: {getattr(l3, 'layer_type', 'N/A')}")
                l3a = [a for a in dir(l3) if not a.startswith("_")]
                lines.append(f"layer3 attrs: {l3a}")
                if hasattr(l3, "self_attn"):
                    sa = l3.self_attn
                    lines.append(f"l3.self_attn: {type(sa).__name__}")
                    sa_a = [a for a in dir(sa) if not a.startswith("_")]
                    lines.append(f"l3.self_attn attrs: {sa_a}")
                if hasattr(l3, "mlp"):
                    mlp = l3.mlp
                    lines.append(f"l3.mlp: {type(mlp).__name__}")
                    mlp_a = [a for a in dir(mlp) if not a.startswith("_")]
                    lines.append(f"l3.mlp attrs: {mlp_a}")
                # Check config
                if hasattr(inner, "config"):
                    cfg = inner.config
                    for k in ["hidden_size", "num_attention_heads",
                              "num_key_value_heads", "head_dim",
                              "num_experts", "num_experts_per_tok"]:
                        lines.append(f"config.{k}: {getattr(cfg, k, 'N/A')}")
        return "\n".join(lines)

    results = llm.llm_engine.apply_model(probe)
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
