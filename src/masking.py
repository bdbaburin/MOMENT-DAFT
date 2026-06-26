import torch

def generate_mask_on_gpu(real_mask: torch.Tensor, patch_len: int = 8, mask_ratio: float = 0.3):
    """
    Генератор масок, полностью соответствующий логике класса Masking в MOMENT.
    
    Аргументы:
        real_mask: Тензор формы [B, 1, L], где 1.0 - валидные данные, 0.0 - паддинг.
        patch_len: Длина одного патча (P). По умолчанию 8.
        mask_ratio: Доля маскируемых патчей (обычно 0.3).
        
    Возвращает:
        input_mask: Маска для механизма Attention [B, L]
        moment_mask: Маска для модели [B, L] (1 - keep, 0 - [MASK])
        loss_mask: Булева маска для расчета MSE по маскированным точкам [B, L]
    """
    B, _, L = real_mask.shape
    num_patches = L // patch_len
    device = real_mask.device
    
    patch_real_mask = real_mask.view(B, num_patches, patch_len)
    
    healthy_patches = patch_real_mask.bool().all(dim=2) # Форма: [B, num_patches]

    rand_tensor = torch.rand(B, num_patches, device=device)
    
    artificial_mask = (rand_tensor < mask_ratio) & healthy_patches # Форма: [B, num_patches]

    moment_patch_mask = healthy_patches & (~artificial_mask)
    
    moment_mask = torch.repeat_interleave(moment_patch_mask, patch_len, dim=1).long() # [B, L]
    loss_mask = torch.repeat_interleave(artificial_mask, patch_len, dim=1).bool()     # [B, L]
    input_mask = real_mask.squeeze(1).long()                                          # [B, L]
    
    # Архитектурная защита: если из-за обилия паддинга в ряду не осталось ни одного 
    # наблюдаемого патча, принудительно открываем первый патч во избежание деления на ноль в RevIN
    safe_guard = (moment_mask.sum(dim=1) == 0)
    if safe_guard.any():
        moment_mask[safe_guard, :patch_len] = 1
        
    return input_mask, moment_mask, loss_mask
