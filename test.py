from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-4B",
    dtype="bfloat16",       # or float16
    tensor_parallel_size=1, # >1 if multi-GPU
    hf_overrides={"architectures": ["MyModelForCausalLM"]},
)

sampling_params = SamplingParams(
    max_tokens=512,
    temperature=0.2,
    top_p=0.95
)

prompts = [
    "The sky is blue because"
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
