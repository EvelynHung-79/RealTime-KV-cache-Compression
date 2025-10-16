import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer
from typing import Optional, Tuple, Union, Dict, List
import logging
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import math

logging.basicConfig(level=logging.INFO)

class CompressedLlamaAttention(LlamaAttention): #繼承 LlamaAttention
    """
    目標是取代 LLaMA 原本的注意力機制 (LlamaAttention)，在計算 Attention 的同時，即時地對 Key-Value (KV) Cache 進行壓縮。

    繼承與擴充：它繼承自 transformers.models.llama.modeling_llama.LlamaAttention，這樣可以重複使用大部分原始的程式碼（如 QKV 投影、RoPE 等），只需專注於修改關鍵部分。
    注入壓縮器 (set_compressor)：它設計了一個 self.compressor 屬性和 set_compressor 方法。這是一種依賴注入 (Dependency Injection) 的設計模式，非常靈活。
    這代表 Attention 層本身不關心「如何」壓縮，只關心「呼叫」壓縮器來完成工作。你的壓縮演算法 (RealTimePrefillCompressor) 可以獨立開發和修改，而不用動到這份程式碼。

    修改 forward 方法：這是最重要的部分。
    - 標準計算：它首先執行標準的 Attention 流程：計算 Q, K, V，應用 RoPE，並計算出原始的 attn_weights。
    - 觸發壓縮：在 if self.compressor is not None 區塊中，它會呼叫 self.compressor.compress_layer_kv_cache。這是執行壓縮的關鍵步驟。
    - 傳遞關鍵資訊：它將 key_states, value_states, attn_weights, input_ids 和 layer_idx 等重要資訊傳遞給壓縮器。
      這完全對應了你設計中的**「Prompt-guided Importance 計算」和「Layer-specific importance weights」**。
    - 處理壓縮後的 KV：
        - 如果壓縮過程減少了 token 數量（Selective Token Propagation），程式碼會用壓縮後的 compressed_key_states 重新計算注意力權重 (compressed_attn_weights)。
        - 如果只是量化（Dynamic Precision Assignment），token 數量不變，則可以直接使用原始的 attn_weights 和 compressed_value_states 計算結果，效率更高。
    - 回傳壓縮結果：當 use_cache=True 時，它回傳的是壓縮過後的 (compressed_key_states, compressed_value_states)，這樣下一輪生成 token 時，
      傳入的 past_key_value 就是已經壓縮過的版本，從而實現記憶體節省。
    """

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.compressor = None  # Will be set externally

        # Ensure compatibility with Hugging Face LlamaAttention
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.head_dim = config.hidden_size // self.num_heads

    def set_compressor(self, compressor):
        """Set the compression module"""
        self.compressor = compressor

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,  # 新增這行
        input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # logging.info(f"[CompressedLlamaAttention] Forwarding at layer {self.layer_idx}")
        bsz, q_len, _ = hidden_states.size()

        # Standard attention computation
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE - 使用內建的函數
        if hasattr(self, 'rotary_emb') and self.rotary_emb is not None:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)


        # Apply RoPE
        # cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Handle past key values
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Normalize attention weights
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Apply real-time compression if compressor is available
        compressed_key_states = key_states
        compressed_value_states = value_states
        compression_info = {}

        # Enter compression block~
        if self.compressor is not None and self.training == False and q_len > 1:  # Only during prefill stage
            try:
                # Reshape for compression (merge batch and head dimensions temporarily)
                k_for_compression = key_states.transpose(1, 2).contiguous()  # [batch, seq_len, heads*head_dim]
                v_for_compression = value_states.transpose(1, 2).contiguous()
                k_for_compression = k_for_compression.view(bsz, -1, self.num_key_value_heads * self.head_dim)
                v_for_compression = v_for_compression.view(bsz, -1, self.num_key_value_heads * self.head_dim)

                # Apply compression
                # 每一層（layer）在 forward 時都會執行一次壓縮（compression）操作。
                # 將這一層的 key/value cache 壓縮，減少記憶體用量或加速運算。
                compressed_k, compressed_v, compression_info = self.compressor.compress_layer_kv_cache(
                    k_for_compression, v_for_compression, attn_weights, 
                    input_ids if input_ids is not None else torch.zeros((bsz, q_len), device=hidden_states.device, dtype=torch.long),
                    self.layer_idx # 追踪當前層
                )

                # Reshape back to attention format
                compressed_seq_len = compressed_k.shape[1]
                compressed_key_states = compressed_k.view(bsz, compressed_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                compressed_value_states = compressed_v.view(bsz, compressed_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

                # Recompute attention with compressed KV
                if compressed_seq_len != key_states.shape[2]:
                    # Adjust query to match compressed sequence length
                    compressed_attn_weights = torch.matmul(
                        query_states, compressed_key_states.transpose(2, 3)
                    ) / math.sqrt(self.head_dim)

                    # Apply mask if needed (truncate to match compressed length)
                    if attention_mask is not None and compressed_seq_len < attention_mask.shape[-1]:
                        compressed_mask = attention_mask[..., :compressed_seq_len]
                        compressed_attn_weights = compressed_attn_weights + compressed_mask

                    compressed_attn_weights = nn.functional.softmax(
                        compressed_attn_weights, dim=-1, dtype=torch.float32
                    ).to(query_states.dtype)

                    attn_output = torch.matmul(compressed_attn_weights, compressed_value_states)
                else:
                    attn_output = torch.matmul(attn_weights, compressed_value_states)

            except Exception as e:
                print(f"Compression failed at layer {self.layer_idx}: {e}")
                # Fallback to original computation
                attn_output = torch.matmul(attn_weights, value_states)
                compressed_key_states = key_states
                compressed_value_states = value_states
        else:
            # Standard attention computation
            attn_output = torch.matmul(attn_weights, value_states)

        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        # Prepare outputs
        outputs = (attn_output,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += ((compressed_key_states, compressed_value_states),)

        return outputs

class CompressedLlamaDecoderLayer(LlamaDecoderLayer):
    """
    目標是作為一個「容器」，將我們客製化的 CompressedLlamaAttention 模組安裝到 LLaMA 的標準 Transformer Decoder 層中，取代原本的 Attention 模組。

    1. 繼承與替換：它繼承自 LlamaDecoderLayer。在 __init__ 方法中，最關鍵的一行是 self.self_attn = CompressedLlamaAttention(config, layer_idx)。它直接用我們寫好的壓縮版 Attention 替換掉了原始的 self_attn。
    2. 橋樑作用 (set_compressor)：它提供了一個 set_compressor 方法，但這個方法只是簡單地呼叫其內部 self.self_attn.set_compressor(compressor)。它像一個橋樑，讓更高層的物件（CompressedLlamaForCausalLM）可以方便地將壓縮器設定到底層的 Attention 模組中。
    3. 確保參數傳遞 (forward)：它的 forward 方法主要工作是確保所有必要的參數（特別是像 cache_position 和我們額外需要的 input_ids）都能被正確地傳遞給底層的 self.self_attn。這裡使用 **kwargs 是一個非常好的實踐，可以確保程式碼對未來 Hugging Face 的更新有更好的相容性。
    """
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = CompressedLlamaAttention(config, layer_idx)
        self.layer_idx = layer_idx

    def set_compressor(self, compressor):
        self.self_attn.set_compressor(compressor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        # 1️⃣ 最重要：確保 hidden_states 是 Tensor
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            input_ids=input_ids,
            **kwargs,
        )

        attn_output = attn_outputs[0]
        self_attn_weights = attn_outputs[1] if output_attentions else None
        present_key_value = attn_outputs[2] if use_cache else None

        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # 2️⃣ 返回 Tensor 而不是 tuple 包裹 Tensor
        outputs = hidden_states
        if output_attentions and use_cache:
            return outputs, self_attn_weights, present_key_value
        elif output_attentions:
            return outputs, self_attn_weights
        elif use_cache:
            return outputs, present_key_value
        else:
            return outputs

class CompressedLlamaForCausalLM(LlamaForCausalLM): # LlamaForCausalLM 是 transformers 裡的類別，用於因果語言模型，而CausalLM 是指模型在生成文本時，只能基於先前的上下文進行預測，不能看到未來的詞彙。
    """
    這是最高層級的完整模型。它的目標是將所有原始的 LlamaDecoderLayer 都替換成我們的 CompressedLlamaDecoderLayer，並提供一個統一的介面來管理壓縮器和獲取壓縮統計數據。

    1. 替換所有層：在 __init__ 中，它使用一個 list comprehension self.model.layers = nn.ModuleList([...]) 來建立一個由 CompressedLlamaDecoderLayer 組成的模型層列表。這確保了模型中的每一層都具備了即時壓縮的能力。
    2. 集中管理壓縮器 (set_compressor)：這個方法遍歷模型中的每一層，並呼叫它們的 set_compressor 方法。這樣，我們只需要呼叫一次 model.set_compressor()，就可以為整個模型配置好壓縮器。
    3. 優雅地擴充 forward：forward 方法的寫法很聰明。它主要還是依賴 super().forward() 來執行大部分工作，但做了兩件重要的事：
        - 傳遞 input_ids 到 CompressedLlamaAttention 中：它確保 input_ids 被保存下來，以便在 CompressedLlamaAttention 中使用。
        - 它在模型的標準輸出 CausalLMOutputWithPast 上附加了 compression_info 屬性。這樣做的好處是，它在不破壞原始輸出結構的情況下，增加了額外的除錯資訊，讓使用者可以輕鬆獲取壓縮統計數據。
    4. 提供 API (get_compression_stats, reset_compression_state)：它提供了清晰的 API 來獲取統計數據和重置狀態，這對於進行實驗和評估非常重要。
    """

    def __init__(self, config):
        super().__init__(config)

        # Replace decoder layers with compressed versions
        self.model.layers = nn.ModuleList([
            CompressedLlamaDecoderLayer(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])

        self.compressor = None
        self.compression_stats = {}

    def set_compressor(self, compressor):
        """Set compressor for all layers"""
        self.compressor = compressor
        for layer in self.model.layers:
            layer.set_compressor(compressor)

    from transformers.modeling_outputs import CausalLMOutputWithPast

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # <-- 接受額外所有 kwargs（例如 cache_position）
    ):
        # 保存 input_ids（若存在）
        if input_ids is not None:
            self._current_input_ids = input_ids
        elif inputs_embeds is not None:
            # 若只有 inputs_embeds，建立 placeholder
            self._current_input_ids = torch.zeros((inputs_embeds.shape[0], inputs_embeds.shape[1]),
                                                dtype=torch.long, device=inputs_embeds.device)

        # 將 **kwargs 傳給 super().forward 以保留 transformers 的行為
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,  # <-- 傳下去，避免 TypeError
        )

        # 確保 outputs 還是 ModelOutput（例如 CausalLMOutputWithPast），然後附加 compression 資訊
        try:
            compression_stats = {}
            if self.compressor is not None:
                compression_stats = self.compressor.get_overall_compression_stats()
            # attach without破壞原本型別
            if hasattr(outputs, '__dict__'):
                setattr(outputs, 'compression_info', compression_stats)
            else:
                # 若 outputs 是 tuple，嘗試包成 CausalLMOutputWithPast
                logits = outputs[0] if isinstance(outputs, (tuple, list)) and len(outputs) > 0 else None
                past = outputs[1] if isinstance(outputs, (tuple, list)) and len(outputs) > 1 else None
                outputs = CausalLMOutputWithPast(logits=logits, past_key_values=past)
                outputs.compression_info = compression_stats
        except Exception as e:
            # 不應該阻斷推論，若附加失敗就忽略
            logging.warning(f"Could not attach compression_info: {e}")

        return outputs


    def get_compression_stats(self) -> Dict:
        """Get comprehensive compression statistics"""
        if self.compressor is not None:
            return self.compressor.get_overall_compression_stats()
        return {}

    def reset_compression_state(self):
        """Reset compression state for new sequence"""
        if self.compressor is not None:
            self.compressor.reset_compression_state()

# Helper function for RoPE (simplified version)
# def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
#     """Apply rotary positional embedding"""
#     # This is a simplified implementation
#     # In practice, you should use the actual RoPE implementation from transformers
#     return q, k


def create_compressed_llama_model(model_name: str, config, device: str = "cuda"):
    """
    Factory function to create compressed LLaMA model

    Args:
        model_name: HuggingFace model name
        config: compression configuration
        device: target device

    Returns:
        Compressed LLaMA model with real-time KV cache compression
    """
    from ..compression.unified_compressor import RealTimePrefillCompressor


    # Load base model configuration
    model_config = LlamaConfig.from_pretrained(model_name)

    # Create compressed model
    model = CompressedLlamaForCausalLM(model_config)

    # Load pre-trained weights (this might need adjustment for the modified layers)
    try:
        base_model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        # Copy weights from base model (skip compression-specific layers)
        model.load_state_dict(base_model.state_dict(), strict=False)
    except Exception as e:
        print(f"Warning: Could not load pre-trained weights: {e}")

    # Initialize compressor
    compressor = RealTimePrefillCompressor(config, model_config)
    model.set_compressor(compressor)

    # Move to device
    model = model.to(device)
    model.eval()  # Set to eval mode for inference

    return model