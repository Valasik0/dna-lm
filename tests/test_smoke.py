"""Základní kontrola, že balíček a klíčové knihovny jdou naimportovat."""


def test_import_dnalm():
    import dnalm

    assert dnalm.__version__


def test_core_libraries():
    import numpy
    import torch
    import transformers

    assert torch.__version__
    assert transformers.__version__
    assert numpy.__version__


def test_modernbert_forward():
    """Mini ModernBERT model projde forward passem na CPU (stejná architektura jako vlastní modely)."""
    import torch
    from transformers import ModernBertConfig, ModernBertForMaskedLM

    config = ModernBertConfig(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
    )
    model = ModernBertForMaskedLM(config).eval()
    input_ids = torch.randint(3, 16, (2, 20))
    with torch.no_grad():
        out = model(input_ids=input_ids)
    assert out.logits.shape == (2, 20, 16)
