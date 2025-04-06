import os
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
import yaml
import logging
import sys
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    PreTrainedTokenizer,
    BitsAndBytesConfig
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProcessor:
    """LLM处理器，用于加载语言模型和生成响应"""

    def __init__(
            self,
            config_path: str = "config/audio_config.yaml",
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化LLM处理器

        Args:
            config_path: 配置文件路径
            device: 设备 ("cuda" 或 "cpu")
        """
        self.device = device

        # 加载配置
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if 'llm' not in config:
                logger.warning("LLM configuration not found in config file, using default")
                config['llm'] = {
                    'model_type': 'auto',
                    'model_name_or_path': 'Qwen/QwQ-32B',  # 默认使用QwQ-32B模型
                    'max_text_length': 512
                }
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using default LLM settings")
            config = {'llm': {
                'model_type': 'auto',
                'model_name_or_path': 'Qwen/QwQ-32B',
                'max_text_length': 512
            }}

        self.llm_config = config['llm']
        self.model_name = self.llm_config['model_name_or_path']
        self.tokenizer_name = self.llm_config.get('tokenizer_name_or_path', self.model_name)
        self.max_text_length = self.llm_config.get('max_text_length', 512)
        self.quantization = self.llm_config.get('quantization', 'auto')

        # 加载分词器和模型
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

        # 特殊令牌
        self._add_special_tokens()

        logger.info(f"Initialized LLMProcessor with model: {self.model_name}")

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """
        加载分词器

        Returns:
            预训练分词器
        """
        try:
            # 检查huggingface-cli缓存目录
            from huggingface_hub.constants import HF_HUB_CACHE
            from transformers.utils import TRANSFORMERS_CACHE
            
            # 首先尝试从HF_HOME环境变量获取缓存目录
            cache_dir = os.getenv("HF_HOME", None)
            if cache_dir is None:
                # 然后尝试从huggingface_hub获取缓存目录
                cache_dir = HF_HUB_CACHE
            if cache_dir is None:
                # 最后尝试从transformers获取缓存目录
                cache_dir = TRANSFORMERS_CACHE
            
            logger.info(f"Loading tokenizer: {self.tokenizer_name}")
            logger.info(f"Using cache directory: {cache_dir}")
            
            tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                trust_remote_code=True,  # 允许远程代码执行，对某些模型（如Qwen）是必需的
                use_fast=False  # 对于一些中文模型，可能需要禁用fast tokenizer
            )

            # 对于没有pad_token的分词器，使用eos_token作为pad_token
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.pad_token = tokenizer.eos_token = "</s>"

            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer '{self.tokenizer_name}': {e}")
            logger.info("Trying fallback tokenizer: gpt2")
            try:
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                return tokenizer
            except Exception as e2:
                logger.error(f"Failed to load fallback tokenizer: {e2}")
                raise

    def _get_gpu_memory(self) -> int:
        """
        获取可用GPU显存大小（GB）
        
        Returns:
            可用GPU显存大小（GB）
        """
        try:
            if not torch.cuda.is_available():
                return 0
            
            # 获取当前设备的总显存和已用显存
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # 转换为GB
            return int(total_memory)
        except Exception as e:
            logger.warning(f"Failed to get GPU memory: {e}, defaulting to 16GB")
            return 16

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """
        根据可用GPU显存选择量化配置
        
        Returns:
            量化配置对象或None
        """
        if self.quantization == 'none' or self.device != 'cuda':
            logger.info("Quantization disabled, using full precision")
            return None
        
        if self.quantization != 'auto':
            # 使用指定的量化级别
            bits = int(self.quantization.replace('int', ''))
            logger.info(f"Using specified quantization: {bits}-bit")
            return self._create_quantization_config(bits)
        
        # 自动选择量化级别
        gpu_memory = self._get_gpu_memory()
        logger.info(f"Detected GPU memory: {gpu_memory}GB")
        
        if gpu_memory >= 64:
            logger.info("Sufficient GPU memory (64GB+), using full precision")
            return None
        elif gpu_memory >= 40:
            logger.info("Using 8-bit quantization for 40GB+ GPU")
            return self._create_quantization_config(8)
        elif gpu_memory >= 16:
            logger.info("Using 4-bit quantization for 16GB+ GPU")
            return self._create_quantization_config(4)
        else:
            logger.info("Limited GPU memory (<16GB), using 4-bit quantization with offloading")
            return self._create_quantization_config(4, offload=True)

    def _create_quantization_config(self, bits: int, offload: bool = False) -> BitsAndBytesConfig:
        """
        创建量化配置
        
        Args:
            bits: 量化位数 (4 或 8)
            offload: 是否启用CPU卸载
            
        Returns:
            BitsAndBytesConfig对象
        """
        if bits == 4:
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                llm_int8_enable_fp32_cpu_offload=offload
            )
        elif bits == 8:
            config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=offload
            )
        else:
            raise ValueError(f"Unsupported quantization bits: {bits}")
        
        return config

    def _load_model(self) -> nn.Module:
        """
        加载语言模型

        Returns:
            预训练模型
        """
        try:
            # 使用预先下载的缓存
            cache_dir = "F:/huggingface_cache"
            
            logger.info(f"Loading model: {self.model_name}")
            logger.info(f"Using cache directory: {cache_dir}")
            logger.info(f"Using device: {self.device}")
            
            # 获取量化配置
            gpu_memory = self._get_gpu_memory()
            logger.info(f"Detected GPU memory: {gpu_memory}GB")
            
            # 始终使用较小的模型进行测试
            if self.model_name == "Qwen/QwQ-32B" and gpu_memory < 48:
                logger.info("For testing purposes, switching to a smaller model: Qwen/Qwen1.5-1.8B-Chat")
                self.model_name = "Qwen/Qwen1.5-1.8B-Chat"
                
            # 定义内存使用策略
            max_memory = None
            if gpu_memory > 0:
                # 为GPU预留10%的显存以避免OOM
                gpu_limit = int(gpu_memory * 0.9) * (1024**3)  # 转为字节
                max_memory = {0: gpu_limit, "cpu": 120 * (1024**3)}  # 120GB CPU内存
                
            # 确定是否需要量化和CPU卸载
            quantization_config = self._get_quantization_config()
            offload = gpu_memory < 24  # 如果GPU显存小于24GB，启用卸载
            
            # 根据是否使用量化调整加载参数
            load_params = {
                "pretrained_model_name_or_path": self.model_name,
                "cache_dir": cache_dir,
                "trust_remote_code": True,
                "device_map": "auto",
                "max_memory": max_memory,
                "low_cpu_mem_usage": True,
                "offload_folder": "offload_models" if offload else None
            }
            
            if quantization_config is not None:
                load_params["quantization_config"] = quantization_config
            else:
                # 如果不使用量化，则使用浮点精度
                load_params["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32
            
            # 加载模型
            model = AutoModelForCausalLM.from_pretrained(**load_params)
            
            # 确认模型设备
            if hasattr(model, "hf_device_map"):
                logger.info(f"Model loaded with device map: {model.hf_device_map}")
            else:
                logger.info(f"Model device: {next(model.parameters()).device}")
                
            return model
        except Exception as e:
            logger.error(f"Failed to load model '{self.model_name}': {e}")
            logger.info("Falling back to a smaller model: gpt2")
            try:
                # 尝试加载更小的模型作为后备选项
                model = AutoModelForCausalLM.from_pretrained(
                    "gpt2",
                    device_map="auto",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                return model
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                raise

    def _add_special_tokens(self):
        """
        添加特殊令牌到分词器
        """
        special_tokens = {
            "additional_special_tokens": []
        }

        # 添加音频标记
        if "<AUDIO>" not in self.tokenizer.additional_special_tokens:
            special_tokens["additional_special_tokens"].append("<AUDIO>")

        # 添加分隔符
        if "<SEP>" not in self.tokenizer.additional_special_tokens:
            special_tokens["additional_special_tokens"].append("<SEP>")

        # 如果有需要添加的特殊令牌
        if special_tokens["additional_special_tokens"]:
            num_added = self.tokenizer.add_special_tokens(special_tokens)
            if num_added > 0:
                # 调整模型词汇表大小
                self.model.resize_token_embeddings(len(self.tokenizer))
                logger.info(f"Added {num_added} special tokens to tokenizer")

    def prepare_inputs(
            self,
            query_embeds: torch.Tensor,
            question: str
    ) -> Dict[str, torch.Tensor]:
        """
        准备模型输入

        Args:
            query_embeds: 查询嵌入 [B, num_query_tokens, hidden_size]
            question: 问题文本

        Returns:
            模型输入字典
        """
        batch_size = query_embeds.size(0)

        # 添加音频标记和问题
        prompts = [f"<AUDIO><SEP>{question}" for _ in range(batch_size)]

        # 分词
        encodings = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt"
        )
        
        # 确保所有张量在正确设备上
        input_dict = {
            "input_ids": encodings["input_ids"].to(self.device),
            "attention_mask": encodings["attention_mask"].to(self.device),
            "query_embeds": query_embeds.to(self.device)
        }

        return input_dict

    def tokenize(
            self,
            texts: Union[str, List[str]],
            padding: str = "max_length",
            truncation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        文本分词

        Args:
            texts: 文本或文本列表
            padding: 填充策略
            truncation: 是否截断

        Returns:
            分词结果
        """
        encodings = self.tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=self.max_text_length,
            return_tensors="pt"
        ).to(self.device)

        return encodings

    def forward(
            self,
            query_embeds: torch.Tensor,
            question: str,
            max_new_tokens: int = 50,
            num_beams: int = 1,
            min_length: int = 1,
            top_p: float = 0.9,
            temperature: float = 0.7,
            repetition_penalty: float = 1.0,
            length_penalty: float = 1.0,
            do_sample: bool = True
    ) -> List[str]:
        """
        执行前向传播并生成回答

        Args:
            query_embeds: 查询嵌入 [B, num_query_tokens, hidden_size]
            question: 问题文本
            max_new_tokens: 最大生成令牌数
            num_beams: 光束搜索数量
            min_length: 最小生成长度
            top_p: top-p采样参数
            temperature: 温度参数
            repetition_penalty: 重复惩罚
            length_penalty: 长度惩罚
            do_sample: 是否使用采样

        Returns:
            生成的回答列表
        """
        # 准备模型输入
        inputs = self.prepare_inputs(query_embeds, question)
        
        answers = []
        
        try:
            with torch.no_grad():
                # 生成标记
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    min_length=min_length,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    do_sample=do_sample,
                    use_cache=True
                )
                
                # 解码生成的标记
                for output_ids in generated_ids:
                    # 获取仅生成的部分（去除提示）
                    input_length = inputs["input_ids"].shape[1]
                    gen_ids = output_ids[input_length:]
                    
                    # 解码
                    gen_text = self.tokenizer.decode(
                        gen_ids, 
                        skip_special_tokens=True, 
                        clean_up_tokenization_spaces=True
                    )
                    
                    answers.append(gen_text)
                    
        except Exception as e:
            logger.error(f"Generation error: {e}")
            
            # 检查是否是CUDA内存不足错误
            if "CUDA out of memory" in str(e):
                logger.warning("CUDA out of memory error. Consider using a lower quantization level.")
                
                # 尝试使用CPU进行生成
                try:
                    logger.info("Attempting generation on CPU...")
                    
                    # 将输入移动到CPU
                    cpu_inputs = {k: v.cpu() for k, v in inputs.items()}
                    
                    # 创建cpu模型副本
                    import copy
                    cpu_model = copy.deepcopy(self.model).cpu()
                    
                    # 使用CPU生成
                    with torch.no_grad():
                        generated_ids = cpu_model.generate(
                            input_ids=cpu_inputs["input_ids"],
                            attention_mask=cpu_inputs["attention_mask"],
                            max_new_tokens=max_new_tokens,
                            num_beams=1,  # 降低束搜索数，以节省内存
                            min_length=min_length,
                            top_p=top_p,
                            temperature=temperature,
                            repetition_penalty=repetition_penalty,
                            do_sample=do_sample,
                            use_cache=True
                        )
                        
                        # 解码生成的标记
                        for output_ids in generated_ids:
                            # 获取仅生成的部分（去除提示）
                            input_length = cpu_inputs["input_ids"].shape[1]
                            gen_ids = output_ids[input_length:]
                            
                            # 解码
                            gen_text = self.tokenizer.decode(
                                gen_ids, 
                                skip_special_tokens=True, 
                                clean_up_tokenization_spaces=True
                            )
                            
                            answers.append(gen_text)
                    
                    # 删除CPU模型副本，释放内存
                    del cpu_model
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                except Exception as cpu_e:
                    logger.error(f"CPU generation also failed: {cpu_e}")
                    answers = ["生成失败，可能是由于内存不足。请尝试使用更高级别的量化或减少输入长度。"]
            else:
                answers = [f"生成错误: {e}"]
            
        return answers


class AudioLLMInterface(nn.Module):
    """音频LLM接口，连接音频特征和LLM"""

    def __init__(
            self,
            config_path: str = "config/audio_config.yaml",
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化音频LLM接口

        Args:
            config_path: 配置文件路径
            device: 设备 ("cuda" 或 "cpu")
        """
        super().__init__()
        self.device = device

        # 加载配置
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if 'audio_llm' not in config:
                logger.warning("AudioLLM configuration not found in config file, using default")
                config['audio_llm'] = {
                    'qformer_hidden_size': 768,
                    'llm_hidden_size': 4096,
                    'cache_dir': 'F:/huggingface_cache'
                }
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using default AudioLLM settings")
            config = {'audio_llm': {
                'qformer_hidden_size': 768,
                'llm_hidden_size': 4096,
                'cache_dir': 'F:/huggingface_cache'
            }}

        self.audio_llm_config = config['audio_llm']
        self.qformer_hidden_size = self.audio_llm_config.get('qformer_hidden_size', 768)
        self.llm_hidden_size = self.audio_llm_config.get('llm_hidden_size', 4096)

        # 创建从QFormer到LLM的投影层
        self.llm_proj = nn.Linear(self.qformer_hidden_size, self.llm_hidden_size)

        # 初始化LLM处理器
        self.llm_processor = LLMProcessor(config_path, device)

        logger.info("Initialized AudioLLMInterface")

    def forward(
            self,
            query_embeds: torch.Tensor,
            question: str,
            max_new_tokens: int = 50,
            num_beams: int = 1,
            min_length: int = 1,
            top_p: float = 0.9,
            temperature: float = 0.7,
            repetition_penalty: float = 1.0,
            length_penalty: float = 1.0,
            do_sample: bool = True
    ) -> List[str]:
        """
        执行前向传播并生成回答

        Args:
            query_embeds: 查询嵌入 [B, num_query_tokens, hidden_size]
            question: 问题文本
            max_new_tokens: 最大生成令牌数
            num_beams: 光束搜索数量
            min_length: 最小生成长度
            top_p: top-p采样参数
            temperature: 温度参数
            repetition_penalty: 重复惩罚
            length_penalty: 长度惩罚
            do_sample: 是否使用采样

        Returns:
            生成的回答列表
        """
        # 投影到LLM空间
        projected_query_embeds = self.llm_proj(query_embeds)
        
        # 使用LLM处理器生成文本
        answers = self.llm_processor.forward(
            query_embeds=projected_query_embeds,
            question=question,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            min_length=min_length,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            do_sample=do_sample
        )
        
        return answers


# 工厂函数，用于创建AudioLLMInterface实例
def create_audio_llm_interface(
        config_path: str = "config/audio_config.yaml",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> AudioLLMInterface:
    """
    创建AudioLLMInterface实例
    
    Args:
        config_path: 配置文件路径
        device: 设备 ("cuda" 或 "cpu")
        
    Returns:
        AudioLLMInterface实例
    """
    return AudioLLMInterface(config_path, device)


# 测试代码
if __name__ == "__main__":
    # 创建配置文件目录（如果不存在）
    os.makedirs("config", exist_ok=True)

    # 如果配置文件不存在，创建一个临时配置
    if not os.path.exists("config/audio_config.yaml"):
        with open("config/audio_config.yaml", "w", encoding="utf-8") as f:
            f.write("""
qformer:
  hidden_size: 768
  num_attention_heads: 12
  num_hidden_layers: 4
  num_query_tokens: 32

llm:
  model_type: 'gpt2'  # 使用较小的模型进行测试
  model_name_or_path: 'gpt2'
  tokenizer_name_or_path: 'gpt2'
  max_text_length: 512
            """)

    try:
        # 创建模型
        llm_interface = create_audio_llm_interface()
        print(f"Created AudioLLMInterface with LLM: {llm_interface.llm_processor.model_name}")

        # 创建测试数据
        batch_size = 1
        num_query_tokens = 32
        hidden_size = 768
        query_embeds = torch.randn(batch_size, num_query_tokens, hidden_size)
        question = "How would you rate the tempo stability of this performance?"

        # 前向传播
        answers = llm_interface(
            query_embeds=query_embeds,
            question=question,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True
        )

        print("\nModel output:")
        print(f"Question: {question}")
        print(f"Answer: {answers[0]}")
    except Exception as e:
        print(f"Error testing LLM interface: {e}")
        print("Note: This test requires downloading a language model, which might take time.")
        print("If the test fails due to model loading issues, you may need to manually download and set up the model.")