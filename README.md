# Lora_with_ResBnLn

model.requires_grad_(False)
lora_layers = warp_with_lora(model, 20)
lora_layers.requires_grad_(True)
