import os

# 全局配置文件
# 控制是否启用大模型进行语义合并
ENABLE_LLM = os.getenv("ENABLE_LLM", "False").lower() in ("true", "1", "t")

# Alfred 数据集路径
ALFRED_DATA_PATH = os.getenv("ALFRED_DATA_PATH", "alfred_data")

# OpenAI API 配置 (如果启用 LLM)
# 优先从环境变量读取，否则使用默认值
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-ZOpDZIXcs0LvE5VtE5D33cE14061425392487d59DeD7Ff71")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://xiaoai.plus/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
