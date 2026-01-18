import config
import difflib
import os
import sys

# 尝试导入 openai 库
try:
    from openai import OpenAI
    # 初始化客户端 (如果配置了 API Key)
    if config.OPENAI_API_KEY and config.OPENAI_API_KEY != "YOUR_API_KEY_HERE":
        client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_API_BASE
        )
    else:
        client = None
except ImportError:
    client = None
    # 只有在启用 LLM 时才打印警告，避免干扰正常运行
    if config.ENABLE_LLM:
        print("Warning: 'openai' module not found. Please install it using `pip install openai`.")
except Exception as e:
    client = None
    if config.ENABLE_LLM:
        print(f"Warning: Failed to initialize OpenAI client: {e}")

# 简单的内存缓存，避免重复查询相同的节点对
_similarity_cache = {}

def are_nodes_semantically_similar(node1_name, node2_name):
    """
    判断两个节点名称在语义上是否相似。
    如果 config.ENABLE_LLM 为 False，直接返回 False。
    如果为 True，使用 LLM 进行判断。
    """
    if not config.ENABLE_LLM:
        return False

    # 预处理名称，移除 ID 后缀 (例如 "-1", "-2")
    n1 = node1_name.split('-')[0].strip()
    n2 = node2_name.split('-')[0].strip()
    
    # 1. 基础规则判断：如果名称完全相同，无需询问 LLM
    if n1 == n2:
        return True
        
    # 如果没有配置好客户端，降级到 difflib
    if not client:
        if config.ENABLE_LLM:
             # 仅打印一次警告或静默降级
             pass
        return _fallback_similarity(n1, n2)

    # 2. 检查缓存
    # 使用排序后的元组作为键，确保 (A, B) 和 (B, A) 命中同一个缓存
    cache_key = tuple(sorted((n1, n2)))
    if cache_key in _similarity_cache:
        return _similarity_cache[cache_key]

    # 3. 调用 LLM
    prompt = f"""
    I am analyzing task graphs for household robots.
    Please determine if the following two action descriptions refer to the semantically equivalent operation, even if the objects or phrasing differ slightly.
    
    Action 1: "{n1}"
    Action 2: "{n2}"
    
    Context: These actions appear in ALFRED-like datasets.
    - "PickupApple" and "PickupApple_1" are equivalent.
    - "SliceApple" and "CutApple" are equivalent.
    - "PickupApple" and "PickupOrange" are NOT equivalent.
    
    Answer only with "YES" if they are semantically similar/equivalent, or "NO" if they are different.
    """

    try:
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert in semantic analysis of robotic plans. Answer concisely."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=5
        )
        content = response.choices[0].message.content.strip().upper()
        
        # 简单解析：只要包含 YES 就认为是
        is_similar = "YES" in content
        
        # 写入缓存
        _similarity_cache[cache_key] = is_similar
        return is_similar
        
    except Exception as e:
        print(f"Error calling LLM for semantic similarity ('{n1}' vs '{n2}'): {e}")
        # 出错时降级到规则判断
        return _fallback_similarity(n1, n2)

def _fallback_similarity(n1, n2):
    """
    降级方案：使用 difflib 进行字符串相似度比较
    """
    seq = difflib.SequenceMatcher(None, n1, n2)
    similarity = seq.ratio()
    return similarity > 0.85

def check_structure_semantically(structure_description, node1, node2):
    """
    询问 LLM 两个子结构是否可以在语义上合并。
    """
    if not config.ENABLE_LLM:
        return False
        
    # 如果需要实现结构层面的语义合并，可以在这里添加类似的 LLM 调用逻辑
    # 目前保持 False，或者根据具体需求实现
    return False
