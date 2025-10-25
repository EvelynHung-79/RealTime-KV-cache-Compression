import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, apply_rotary_pos_emb
from typing import Optional, Tuple, Union, Dict, List
import logging
import math

# Import new components
from ..compression.streaming_quantization import StreamingStatisticsManager, quantize_chunk, dequantize_symmetric
from configs.base_config import CompressionConfig

logging.basicConfig(level=logging.INFO)

# --- Helper function for chunking ---
def iter_chunks(tensor: torch.Tensor, dim: int, chunk_size: int):
    tensor_len = tensor.shape[dim]
    for i in range(0, tensor_len, chunk_size):
        yield tensor.narrow(dim, i, min(chunk_size, tensor_len - i)), i # Yield chunk and start index

# --- Custom KV Cache (Example - Needs Refinement) ---
# A simple placeholder. Real implementation might need more sophisticated
# handling of mixed precision, appending chunks, and indexing.
class QuantizedKVCache:
    def __init__(self):
        self.key_cache = {} # layer_idx -> List[(quantized_chunk, scale)]
        self.value_cache = {} # layer_idx -> List[(quantized_chunk, scale)]
        self.current_length = 0

    def append(self, layer_idx: int, key_quant: torch.Tensor, key_scale: torch.Tensor,
               value_quant: torch.Tensor, value_scale: torch.Tensor):
        if layer_idx not in self.key_cache:
            self.key_cache[layer_idx] = []
            self.value_cache[layer_idx] = []
        # Store quantized tensor and its scale
        self.key_cache[layer_idx].append((key_quant.cpu(), key_scale.cpu())) # Move to CPU to save GPU RAM?
        self.value_cache[layer_idx].append((value_quant.cpu(), value_scale.cpu()))
        # Assume all chunks have same length for simplicity here
        if layer_idx == 0: # Only update length once per chunk step
             self.current_length += key_quant.shape[2] # B, H, C_len, D

    def get_full_cache(self, layer_idx: int, device: torch.device) -> Optional[Tuple[List[Tuple], List[Tuple]]]:
        # Returns list of (quant_chunk, scale) tuples for key and value
        k = [(qc.to(device), qs.to(device)) for qc, qs in self.key_cache.get(layer_idx, [])]
        v = [(vc.to(device), vs.to(device)) for vc, vs in self.value_cache.get(layer_idx, [])]
        if not k:
            return None
        return k, v

    def get_sequence_length(self):
        return self.current_length


class CompressedLlamaAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None,
                 compression_config: Optional[CompressionConfig] = None, # Pass compression config
                 stats_manager: Optional[StreamingStatisticsManager] = None): # Pass stats manager
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.compression_config = compression_config
        self.stats_manager = stats_manager
        # self.quantizer = RealTimeQuantizer() # Functions are used directly now

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        # Ensure num_key_value_heads is correctly obtained from LlamaConfig
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.head_dim = config.hidden_size // self.num_heads

        # Flag to indicate prefill vs decode stage (simple check)
        self.is_prefill = True

        # Custom cache object if needed, or modify past_key_value handling
        # self.custom_cache = None # Example


    # No set_compressor needed, dependencies injected at init

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None, # May need custom handling
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None, # HF specific, may need adaptation
        # input_ids: Optional[torch.Tensor] = None, # Not used in this version
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()
        using_custom_cache = isinstance(past_key_value, QuantizedKVCache) # Example check

        # --- Determine Stage (Prefill vs Decode) ---
        # Simple heuristic: if q_len > 1 and no complex past_key_value, assume prefill
        # A more robust method might involve explicit state management
        current_seq_len = 0
        if past_key_value is not None:
             if using_custom_cache:
                 current_seq_len = past_key_value.get_sequence_length()
             elif isinstance(past_key_value, tuple) and past_key_value[0] is not None:
                 current_seq_len = past_key_value[0].shape[2] # B, H, L, D
        
        # If adding new tokens significantly increases length, assume prefill
        self.is_prefill = (q_len > 1 and current_seq_len < self.compression_config.chunk_size) or current_seq_len == 0


        # --- Prefill Stage with Chunking and Quantization ---
        if self.is_prefill and self.compression_config and self.stats_manager and q_len > 1:
            logging.debug(f"[L{self.layer_idx}] Prefill stage detected (q_len={q_len}, cur_len={current_seq_len})")
            
            # Initialize or get custom cache
            if not using_custom_cache:
                current_cache = QuantizedKVCache()
                # If there's existing non-quantized cache, handle conversion/transfer (complex)
            else:
                current_cache = past_key_value

            # --- Process input hidden_states chunk by chunk ---
            all_attn_outputs = []
            accumulated_pos = current_seq_len

            # Calculate Query states once (or chunk by chunk if memory is tight)
            query_states = self.q_proj(hidden_states)
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, L, D]


            for chunk_hs, start_idx in iter_chunks(hidden_states, dim=1, chunk_size=self.compression_config.chunk_size):
                chunk_len = chunk_hs.shape[1]
                chunk_pos_ids = position_ids[:, start_idx : start_idx + chunk_len] if position_ids is not None else None
                chunk_token_indices = torch.arange(accumulated_pos, accumulated_pos + chunk_len, device=hidden_states.device)

                # 1. Calculate K, V for the chunk
                key_chunk = self.k_proj(chunk_hs) #[B, C_len, H*D]
                value_chunk = self.v_proj(chunk_hs)
                key_chunk = key_chunk.view(bsz, chunk_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) #[B, H, C_len, D]
                value_chunk = value_chunk.view(bsz, chunk_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

                # 2. Key Quantization (Pre-RoPE)
                self.stats_manager.update(self.layer_idx, 'key', key_chunk)
                # quantize_chunk returns (quantized_chunk [int8], scale [float])
                quantized_key_chunk, key_scale = quantize_chunk(
                    key_chunk, self.layer_idx, 'key', chunk_token_indices,
                    self.stats_manager, self.compression_config
                )
                # Store quantized key + scale

                # 3. Apply RoPE (Needs high-precision K/V)
                # Option A: Use original K/V
                # Option B: Temp dequantize K (introduces overhead)
                temp_dequant_key_chunk = dequantize_symmetric(quantized_key_chunk, key_scale)
                cos, sin = self.rotary_emb(value_chunk, seq_len=accumulated_pos + chunk_len) # RoPE needs full seq length context
                # Apply RoPE relative to position: Use chunk_pos_ids
                # query_chunk_rope, key_chunk_rope = apply_rotary_pos_emb(query_states[:, :, start_idx:start_idx+chunk_len, :], temp_dequant_key_chunk, cos, sin, chunk_pos_ids)
                # value_chunk_rope = apply_rotary_pos_emb_to_value(value_chunk, cos, sin, chunk_pos_ids) # If RoPE applied to V too
                
                # --- Simplified RoPE application (check compatibility) ---
                # Assuming RoPE can be applied chunk-wise based on absolute position
                # This might need adjustment based on specific RoPE implementation
                query_chunk_for_rope = query_states[:, :, start_idx : start_idx + chunk_len, :]
                cos_chunk, sin_chunk = self.rotary_emb(value_chunk, seq_len = chunk_len) # Does RoPE depend only on chunk length? Check!
                query_chunk_rope, key_chunk_rope = apply_rotary_pos_emb(query_chunk_for_rope, temp_dequant_key_chunk, cos_chunk, sin_chunk, chunk_pos_ids)
                value_chunk_rope = value_chunk # Assuming RoPE not applied to Value, adjust if needed

                # 4. Value Quantization (Post-RoPE)
                self.stats_manager.update(self.layer_idx, 'value', value_chunk_rope)
                quantized_value_chunk, value_scale = quantize_chunk(
                    value_chunk_rope, self.layer_idx, 'value', chunk_token_indices,
                    self.stats_manager, self.compression_config
                )
                # Store quantized value + scale

                # Append to cache AFTER processing the chunk
                current_cache.append(self.layer_idx, quantized_key_chunk, key_scale, quantized_value_chunk, value_scale)


                # 5. Attention Calculation for the current chunk
                # Retrieve ALL past keys/values (dequantized)
                past_k_tuples, past_v_tuples = current_cache.get_full_cache(self.layer_idx, hidden_states.device)
                
                # Dequantize all keys/values needed for attention calculation up to current point
                # This is potentially expensive - consider if calculation can use quantized values directly (needs QAT)
                all_keys_high_precision = [dequantize_symmetric(qk, qs) for qk, qs in past_k_tuples]
                all_values_high_precision = [dequantize_symmetric(qv, vs) for qv, vs in past_v_tuples]

                if all_keys_high_precision:
                    full_k_for_attn = torch.cat(all_keys_high_precision, dim=2) # Concatenate along sequence dim
                    full_v_for_attn = torch.cat(all_values_high_precision, dim=2)
                    
                    # Apply RoPE to the full dequantized K for attention calculation consistency? Or assume stored versions are post-RoPE?
                    # Let's assume stored K is pre-RoPE, V is post-RoPE based on quantization step. Re-apply RoPE.
                    # This implies storing pre-RoPE K quant, post-RoPE V quant.
                    # --- Recalculating RoPE on full dequantized K (for consistency) ---
                    # cos_full, sin_full = self.rotary_emb(full_v_for_attn, seq_len=current_cache.get_sequence_length())
                    # _, full_k_for_attn_rope = apply_rotary_pos_emb(None, full_k_for_attn, cos_full, sin_full, position_ids[:, :current_cache.get_sequence_length()])
                    
                    # --- Assuming stored V is post-RoPE, and stored K needs RoPE applied after dequant ---
                    # Need accurate position_ids for the cached sequence
                    cached_len = full_k_for_attn.shape[2]
                    cached_pos_ids = position_ids[:, :cached_len] if position_ids is not None else None
                    cos_cache, sin_cache = self.rotary_emb(full_v_for_attn, seq_len=cached_len) # Use Value shape for seq_len context
                    _, full_k_for_attn_rope = apply_rotary_pos_emb(None, full_k_for_attn, cos_cache, sin_cache, cached_pos_ids)


                    # Calculate attention scores for the current query chunk against ALL keys
                    # query_chunk_rope: [B, H, C_len, D]
                    # full_k_for_attn_rope: [B, H, Full_L, D]
                    attn_weights_chunk = torch.matmul(query_chunk_rope, full_k_for_attn_rope.transpose(2, 3)) / math.sqrt(self.head_dim)

                    # Apply mask for the current chunk vs full history
                    if attention_mask is not None:
                         # Mask shape needs careful handling for chunk vs full history
                         mask_slice = attention_mask[:, :, start_idx : start_idx + chunk_len, :full_k_for_attn_rope.shape[2]]
                         attn_weights_chunk = attn_weights_chunk + mask_slice


                    attn_weights_chunk = nn.functional.softmax(attn_weights_chunk, dim=-1, dtype=torch.float32).to(query_states.dtype)
                    # Output calculation using current weights and FULL high-precision values
                    # attn_output_chunk: [B, H, C_len, D]
                    attn_output_chunk = torch.matmul(attn_weights_chunk, full_v_for_attn)

                    all_attn_outputs.append(attn_output_chunk)
                    
                else: # First chunk case (should not happen if cache init handled)
                     # Handle self-attention within the first chunk if necessary
                     logging.warning(f"Layer {self.layer_idx}: No past KV found for chunk starting at {start_idx}")
                     # Simplified: Output zeros or skip? For prefill, context is needed. Error likely.
                     # Assuming first chunk already added to cache before this loop starts or handled separately.
                     pass # Should ideally calculate self-attention within chunk

                accumulated_pos += chunk_len
                if output_attentions: pass # Store attn_weights_chunk if needed


            # Concatenate outputs from all chunks
            attn_output = torch.cat(all_attn_outputs, dim=2) # Cat along sequence dim [B, H, q_len, D]


            # Reshape output
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)

            # --- Prepare outputs for Prefill stage ---
            final_outputs = (attn_output,)
            if output_attentions:
                # Need to handle chunked attention weights storage/concatenation
                final_outputs += (None,) # Placeholder
            if use_cache:
                # Return the custom cache object
                final_outputs += (current_cache,)

            return final_outputs


        # --- Decode Stage (Using Quantized Cache) ---
        else:
             logging.debug(f"[L{self.layer_idx}] Decode stage detected (q_len={q_len}, cur_len={current_seq_len})")
             # Assume past_key_value is our QuantizedKVCache or compatible format
             
             # Standard QKV calculation for the single new token (q_len=1)
             query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
             key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
             value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

             # Apply RoPE to the new Q, K, V
             cos, sin = self.rotary_emb(value_states, seq_len=current_seq_len + q_len)
             # Use the correct positions (often just the next position id)
             query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
             # Apply RoPE to value_states if needed


             # --- Append new K/V (Quantized) to cache ---
             if use_cache and using_custom_cache: # Check if using our cache
                current_cache = past_key_value
                new_token_indices = torch.arange(current_seq_len, current_seq_len + q_len, device=hidden_states.device)

                # Quantize the new key_states (pre-RoPE version needed?) - Simpler: quantize post-RoPE for decode
                # For decode stage, simpler to quantize POST-RoPE key/value
                self.stats_manager.update(self.layer_idx, 'key', key_states) # Using post-RoPE K for stats update in decode
                quantized_key_new, key_scale_new = quantize_chunk(
                    key_states, self.layer_idx, 'key', new_token_indices,
                    self.stats_manager, self.compression_config)

                self.stats_manager.update(self.layer_idx, 'value', value_states) # Using post-RoPE V
                quantized_value_new, value_scale_new = quantize_chunk(
                    value_states, self.layer_idx, 'value', new_token_indices,
                    self.stats_manager, self.compression_config)

                # Append to the custom cache
                current_cache.append(self.layer_idx, quantized_key_new, key_scale_new, quantized_value_new, value_scale_new)
                next_cache = current_cache # Return updated cache object
             else:
                next_cache = None # Or handle standard cache format


             # --- Attention Calculation using FULL Dequantized Cache + New Token ---
             if using_custom_cache:
                past_k_tuples, past_v_tuples = current_cache.get_full_cache(self.layer_idx, hidden_states.device)
                
                # Dequantize ALL keys/values from cache
                all_keys_high_precision = torch.cat([dequantize_symmetric(qk, qs) for qk, qs in past_k_tuples], dim=2)
                all_values_high_precision = torch.cat([dequantize_symmetric(qv, vs) for qv, vs in past_v_tuples], dim=2)
                 
                # Concatenate with the current high-precision key/value (post-RoPE)
                # Note: past K/V retrieved from cache might be pre/post RoPE depending on prefill logic. Assume Consistency needed.
                # If cache stores PRE-RoPE K, apply RoPE after dequant. If POST-RoPE V, use directly.
                # Simplification for Decode: Assume cache K/V are ready post-RoPE (adjust prefill if needed)
                full_k_for_attn = all_keys_high_precision # Assuming cache K is post-RoPE after dequant
                full_v_for_attn = all_values_high_precision # Assuming cache V is post-RoPE after dequant
                
             elif isinstance(past_key_value, tuple): # Standard cache format (assumed high precision)
                  full_k_for_attn = torch.cat([past_key_value[0], key_states], dim=2)
                  full_v_for_attn = torch.cat([past_key_value[1], value_states], dim=2)
                  if use_cache and not using_custom_cache: # Update standard cache
                      next_cache = (full_k_for_attn, full_v_for_attn)

             else: # No cache
                  full_k_for_attn = key_states
                  full_v_for_attn = value_states
                  if use_cache and not using_custom_cache:
                      next_cache = (full_k_for_attn, full_v_for_attn)


             # --- Standard Attention Calculation ---
             attn_weights = torch.matmul(query_states, full_k_for_attn.transpose(2, 3)) / math.sqrt(self.head_dim)
             if attention_mask is not None:
                 attn_weights = attn_weights + attention_mask # Mask should cover full length
             attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
             attn_output = torch.matmul(attn_weights, full_v_for_attn)


             # Reshape output
             attn_output = attn_output.transpose(1, 2).contiguous()
             attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
             attn_output = self.o_proj(attn_output)

             # Prepare outputs for Decode stage
             final_outputs = (attn_output,)
             if output_attentions:
                 final_outputs += (attn_weights,) # Use the calculated weights
             if use_cache:
                 final_outputs += (next_cache,) # Return updated cache (custom or standard)

             return final_outputs



class CompressedLlamaDecoderLayer(LlamaDecoderLayer):
    """Integrates the modified attention layer."""
    def __init__(self, config: LlamaConfig, layer_idx: int,
                 compression_config: Optional[CompressionConfig] = None, # Pass configs down
                 stats_manager: Optional[StreamingStatisticsManager] = None):
        super().__init__(config, layer_idx)
        # Replace attention layer
        self.self_attn = CompressedLlamaAttention(config, layer_idx, compression_config, stats_manager)
        self.layer_idx = layer_idx
        # No set_compressor needed


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None, # Can be standard tuple or custom cache
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None, # Pass through
        # input_ids: Optional[torch.Tensor] = None, # Removed, not used here
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Pass arguments to the modified attention layer
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        attn_output = attn_outputs[0]
        self_attn_weights = attn_outputs[1] if output_attentions else None
        present_key_value = attn_outputs[2] if use_cache else None # This will be the updated cache (custom or standard)

        hidden_states = residual + attn_output # Add dropout if needed

        # Rest of the layer remains the same
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # --- Correct return tuple structure ---
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,) # Append the cache state

        # If only hidden_states is needed (no attn/cache), return just the tensor
        if not output_attentions and not use_cache:
             return hidden_states
        else:
             # Important: The number of elements in the returned tuple MUST match
             # what the higher-level model expects based on output_attentions/use_cache flags.
             # Ensure the length matches `transformers.models.llama.modeling_llama.LlamaDecoderLayer`'s return signature.
             # Example: If use_cache=True, output_attentions=False, expects (hidden_states, present_key_value)
             # Adjust based on exact signature. The code above assumes a flexible return.
             # Let's refine based on typical HF return: (hidden_states,) + (attn_weights,) if output_attentions + (present_key_value,) if use_cache
             return outputs # The tuple structure handles this


class CompressedLlamaForCausalLM(LlamaForCausalLM):
    """LLaMA model with integrated streaming quantization."""

    def __init__(self, config: LlamaConfig, compression_config: CompressionConfig): # Pass compression_config
        super().__init__(config)
        self.compression_config = compression_config

        # Instantiate the statistics manager ONCE for the whole model
        self.stats_manager = StreamingStatisticsManager(compression_config)

        # Replace decoder layers with compressed versions, injecting dependencies
        self.model.layers = nn.ModuleList([
            CompressedLlamaDecoderLayer(config, layer_idx, compression_config, self.stats_manager)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # No set_compressor needed

    # Keep forward signature compatible with Hugging Face LlamaForCausalLM
    # Remove internal _current_input_ids logic if not strictly needed
    from transformers.modeling_outputs import CausalLMOutputWithPast

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None, # This might be List[QuantizedKVCache] now
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None, # Pass down
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # --- Handle potential custom cache format in past_key_values ---
        # The underlying model expects a specific tuple format for past_key_values.
        # If we use a custom QuantizedKVCache object, we might need to:
        # 1. Pass it directly if the modified DecoderLayer handles it.
        # 2. Convert it back/forth if the HF internals require the tuple format (more complex).
        # Assuming option 1 for now (DecoderLayer handles QuantizedKVCache).

        # Decoder Layers
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values, # Pass directly
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float() # Cast to float

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:] # outputs[1:] contains cache, attentions, hidden_states
            return (loss,) + output if loss is not None else output

        # Attach stats if returning dict
        compression_stats = self.get_compression_stats()

        # The 'past_key_values' in the output might be our custom cache object
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if use_cache else None, # Pass the cache state from decoder output
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            # Add custom stats (non-standard HF output)
            # compression_info=compression_stats # Can't directly add here, maybe return separately or log
        )


    # Keep get_compression_stats and reset_compression_state, delegate to manager
    def get_compression_stats(self) -> Dict:
        """Get statistics from the streaming manager."""
        # TODO: Enhance stats_manager to provide meaningful summary stats
        # For now, maybe just return outlier counts or estimated savings
        if hasattr(self, 'stats_manager'):
           # Example: count outliers
           key_outliers = self.stats_manager.key_is_outlier.sum().item()
           val_outliers = self.stats_manager.value_is_outlier.sum().item()
           return {"key_outliers_detected": key_outliers, "value_outliers_detected": val_outliers}
        return {}

    def reset_compression_state(self):
        """Reset streaming statistics."""
        if hasattr(self, 'stats_manager'):
            self.stats_manager.reset()
            # Reset prefill flag in attention layers? Needs careful state management.
            for layer in self.model.layers:
                 if hasattr(layer.self_attn, 'is_prefill'):
                     layer.self_attn.is_prefill = True # Reset prefill state
            logging.info("Reset compression statistics and prefill state.")



def create_compressed_llama_model(model_name: str, compression_config: CompressionConfig, device: str = "cuda"): # Pass compression_config
    """
    Factory function modified for the new architecture.
    """
    # Load base model configuration
    model_config = LlamaConfig.from_pretrained(model_name)

    # --- IMPORTANT: Inject compression_config into LlamaConfig if needed ---
    # Some arguments might need to be passed down through LlamaConfig.
    # Alternatively, pass compression_config directly during layer init. (Chosen approach)

    # Create compressed model, passing the compression config
    model = CompressedLlamaForCausalLM(model_config, compression_config)

    # Load pre-trained weights
    # This should still work as layer names match, only internal logic changed
    try:
        logging.info(f"Loading base weights from {model_name}...")
        base_model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        # Copy weights
        model.load_state_dict(base_model.state_dict(), strict=False)
        del base_model # Free memory
        logging.info("Base weights loaded.")
    except Exception as e:
        logging.warning(f"Could not load pre-trained weights: {e}. Model will be randomly initialized.")

    # Compressor object removed, stats_manager is part of the model now
    # model.set_compressor(...) # Removed

    model = model.to(device)
    model.eval()
    logging.info(f"Compressed LLaMA model created and moved to {device}.")
    return model