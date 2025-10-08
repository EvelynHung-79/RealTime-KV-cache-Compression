import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer
from typing import Optional, Tuple, Union, Dict, List
import logging

logging.basicConfig(level=logging.INFO)

class CompressedLlamaAttention(LlamaAttention):
    """
    Modified LlamaAttention with real-time KV cache compression
    """

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.compressor = None  # Will be set externally

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
        input_ids: Optional[torch.Tensor] = None,  # Added for compression
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    logging.info(f"[CompressedLlamaAttention] Forwarding at layer {self.layer_idx}")
        bsz, q_len, _ = hidden_states.size()

        # Standard attention computation
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

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

        if self.compressor is not None and self.training == False:  # Only during inference
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
    Modified LlamaDecoderLayer with compressed attention
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        # Replace attention with compressed version
        self.self_attn = CompressedLlamaAttention(config, layer_idx)
        self.layer_idx = layer_idx

    def set_compressor(self, compressor):
        """Set compressor for the attention layer"""
        self.self_attn.set_compressor(compressor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        input_ids: Optional[torch.Tensor] = None,  # Added for compression
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention with compression
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            input_ids=input_ids,
        )
        hidden_states = residual + hidden_states

        # Feed Forward
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class CompressedLlamaForCausalLM(LlamaForCausalLM):
    """
    LlamaForCausalLM with real-time KV cache compression
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
    ):

        # Store input_ids for compression
        self._current_input_ids = input_ids

        # Forward pass with compression
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
        )

        # Add compression statistics to output
        if self.compressor is not None:
            compression_stats = self.compressor.get_overall_compression_stats()
            if hasattr(outputs, 'compression_stats'):
                outputs.compression_stats = compression_stats
            else:
                # Add as custom attribute
                outputs.compression_info = compression_stats

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
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Apply rotary positional embedding"""
    # This is a simplified implementation
    # In practice, you should use the actual RoPE implementation from transformers
    return q, k


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