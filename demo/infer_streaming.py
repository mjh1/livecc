"""
Extended LiveCC inference that supports both file-based and streaming video
"""
import torch
import functools
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers.generation.logits_process import LogitsProcessor

from livecc_utils import prepare_multiturn_multimodal_inputs_for_generation
from livecc_utils.video_process_patch import get_smart_resized_video_reader, FRAME_FACTOR
from torchvision import transforms


class ThresholdLogitsProcessor(LogitsProcessor):
    def __init__(self, eos_token_id: int, threshold: float = 0.9, threshold_step: float = 0.0):
        self.count = 0
        self.eos_token_id = eos_token_id
        self.threshold = threshold
        self.threshold_step = threshold_step

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        current_threshold = self.threshold + self.threshold_step * self.count
        eos_probs = torch.nn.functional.softmax(scores, dim=-1)[0, self.eos_token_id]
        if eos_probs > current_threshold:
            scores[:, :] = -float("inf")
            scores[:, self.eos_token_id] = 0
        self.count += 1
        return scores


class LiveCCStreamingInfer:
    """Extended inference class that supports streaming video"""
    
    VIDEO_PLAY_END = object()
    VIDEO_PLAY_CONTINUE = object()
    fps = 2
    initial_fps_frames = 6
    streaming_fps_frames = 2
    initial_time_interval = initial_fps_frames / fps
    streaming_time_interval = streaming_fps_frames / fps
    frame_time_interval = 1 / fps

    def __init__(self, model_path: str = None, device: str = None):
        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", 
            device_map=device, 
            attn_implementation='flash_attention_2' if 'cuda' in device else None
        )
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
        self.streaming_eos_token_id = self.processor.tokenizer(' ...').input_ids[-1]
        self.model.prepare_inputs_for_generation = functools.partial(
            prepare_multiturn_multimodal_inputs_for_generation, self.model
        )
        
        self._cached_video_readers_with_hw = {}

    def get_smart_resized_clip_from_buffer(
        self,
        frame_buffer,
        timestamps: torch.Tensor,
        video_pts: torch.Tensor,
        video_pts_index_from: int = 0,
    ):
        """Get frames from frame buffer instead of video file"""
        
        # Add extra timestamp for alignment
        timestamps = torch.cat([timestamps, timestamps[-1:] + 1 / self.fps])
        
        clip_idxs = []
        clip_frames = []
        
        for timestamp in timestamps:
            # Find the closest frame to this timestamp
            while video_pts_index_from < len(video_pts) and video_pts[video_pts_index_from] < timestamp:
                video_pts_index_from += 1
            if video_pts_index_from >= len(video_pts):
                break
            clip_idxs.append(video_pts_index_from)
        
        # Ensure clip length is multiple of FRAME_FACTOR
        while len(clip_idxs) % FRAME_FACTOR != 0 and len(clip_idxs) > 0:
            clip_idxs = clip_idxs[:-1]
            timestamps = timestamps[:-1]
        
        if not clip_idxs or len(clip_idxs) == 0:
            return None, timestamps, clip_idxs
        
        # Get frames from buffer
        with frame_buffer.frame_lock:
            frames_list = list(frame_buffer.frames)
            for idx in clip_idxs:
                if idx < len(frames_list):
                    _, frame = frames_list[idx]
                    clip_frames.append(frame)
        
        if not clip_frames:
            return None, timestamps, clip_idxs
        
        # Stack frames: [T, H, W, C] -> [T, C, H, W]
        clip = torch.stack([torch.from_numpy(f) for f in clip_frames]).permute(0, 3, 1, 2).float()
        
        return clip, timestamps, clip_idxs

    @torch.inference_mode()
    def live_cc_streaming(
        self,
        message: str,
        state: dict,
        max_pixels: int = 384 * 28 * 28,
        default_query: str = 'Please describe the video.',
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        streaming_eos_base_threshold: float = None,
        streaming_eos_threshold_step: float = None,
        use_kv_cache: bool = False,  # Set to False for independent descriptions
        **kwargs,
    ):
        """
        Process streaming video from frame buffer
        
        state: dict with keys:
            frame_buffer: LiveFrameBuffer instance
            video_timestamp: float, current video timestamp
            last_timestamp: float, last processed video timestamp
            last_video_pts_index: int, last processed video frame index
            video_pts: torch.Tensor, all available timestamps
            last_history: list, last processed history
        """
        # 1. Get streaming state
        video_timestamp = state.get('video_timestamp', 0)
        last_timestamp = state.get('last_timestamp', -1 / self.fps)
        frame_buffer = state.get('frame_buffer')
        
        if frame_buffer is None:
            return
        
        video_pts = state.get('video_pts')
        if video_pts is None or len(video_pts) == 0:
            return
        
        resized_height = frame_buffer.resized_height
        resized_width = frame_buffer.resized_width
        
        if resized_height is None or resized_width is None:
            return
        
        last_video_pts_index = state.get('last_video_pts_index', -1)
        
        # 2. Determine which frames to process
        initialized = last_timestamp >= 0
        if not initialized:
            video_timestamp = max(video_timestamp, self.initial_time_interval)
        
        if video_timestamp <= last_timestamp + self.frame_time_interval:
            return
        
        timestamps = torch.arange(
            last_timestamp + self.frame_time_interval,
            video_timestamp,
            self.frame_time_interval
        )
        
        # 3. Fetch frames from buffer
        clip, clip_timestamps, clip_idxs = self.get_smart_resized_clip_from_buffer(
            frame_buffer, timestamps, video_pts,
            video_pts_index_from=last_video_pts_index + 1
        )
        
        if clip is None or len(clip) == 0:
            return
        
        # 4. Prepare conversation
        conversation = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": None,  # Placeholder, will pass directly to processor
                    "resized_height": resized_height,
                    "resized_width": resized_width,
                    "fps": self.fps,
                },
            ]
        }]
        
        # Add query text
        if message and message.strip():
            conversation[0]['content'].append({"type": "text", "text": message})
        else:
            conversation[0]['content'].append({"type": "text", "text": default_query})
        
        # 5. Build text prompt
        texts = self.processor.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True, 
            return_tensors='pt'
        )
        
        # Get past state (only if using KV cache)
        past_ids = state.get('past_ids', None) if use_kv_cache else None
        past_key_values = state.get('past_key_values', None) if use_kv_cache else None
        
        # For continuing conversations, adjust the prompt
        if past_ids is not None and initialized and use_kv_cache:
            # Remove system prompt that's already in past_ids
            texts = '<|im_end|>\n' + texts.split('<|im_end|>\n', 1)[-1]
        
        # 6. Process inputs - pass video clip directly
        inputs = self.processor(
            text=texts,
            images=None,
            videos=[clip],  # Pass tensor directly as a list
            return_tensors="pt",
            return_attention_mask=False
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # 7. Concatenate with past IDs if continuing
        if past_ids is not None and initialized and use_kv_cache:
            inputs['input_ids'] = torch.cat([past_ids, inputs['input_ids']], dim=1)
        
        # 8. Streaming generation logic
        logits_processor = None
        if streaming_eos_base_threshold is not None:
            logits_processor = [
                ThresholdLogitsProcessor(
                    self.streaming_eos_token_id,
                    streaming_eos_base_threshold,
                    streaming_eos_threshold_step
                )
            ]
        
        # 9. Generate
        outputs = self.model.generate(
            **inputs,
            past_key_values=past_key_values,
            return_dict_in_generate=True,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            logits_processor=logits_processor,
            max_new_tokens=128,  # Increased to allow full descriptions
            pad_token_id=self.model.config.eos_token_id,
            **kwargs
        )
        
        # 10. Decode response
        response = self.processor.decode(
            outputs.sequences[0, inputs['input_ids'].size(1):],
            skip_special_tokens=True
        )
        
        # 11. Update state
        start_timestamp = clip_timestamps[0].item()
        stop_timestamp = clip_timestamps[-1].item()
        
        # Only save KV cache if enabled
        if use_kv_cache:
            state['past_key_values'] = outputs.past_key_values
            state['past_ids'] = outputs.sequences[:, :-1]
        else:
            # Clear any existing cache to ensure fresh descriptions
            state['past_key_values'] = None
            state['past_ids'] = None
        
        state['last_timestamp'] = stop_timestamp
        state['last_video_pts_index'] = clip_idxs[-1]
        
        yield (start_timestamp, stop_timestamp), response, state

