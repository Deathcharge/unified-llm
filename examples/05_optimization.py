"""Example 5: Optimization Techniques"""

from unified_llm.providers import AdvancedQuantization, KVCacheManager

def example_quantization():
    """Demonstrate model quantization."""
    quantizer = AdvancedQuantization(
        method="int8",
        bits=8,
        group_size=128
    )
    
    print("Quantization Configuration:")
    print(f"Method: INT8")
    print(f"Bits: 8")
    print(f"Group Size: 128")
    print()

def example_kv_cache():
    """Demonstrate KV cache management."""
    cache = KVCacheManager()
    
    # Store some values
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    
    # Retrieve values
    val1 = cache.get("key1")
    val2 = cache.get("key2")
    
    print("KV Cache Example:")
    print(f"Retrieved: key1={val1}, key2={val2}")
    print()

if __name__ == "__main__":
    print("=== Optimization Examples ===\n")
    try:
        example_quantization()
        example_kv_cache()
    except Exception as e:
        print(f"Example failed: {e}")
