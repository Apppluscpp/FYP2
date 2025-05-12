# inference_app.py
import streamlit as st
import torch
from datetime import datetime
from training.llama_training_speedup_utils import (
    Tokenizer, generate_v2, text_to_token_ids, token_ids_to_text
)
from training.parameters import LLAMA32_CONFIG_1B, llama_generation_parameters
from training.llama_structure import Llama3Model

# --- Load model ---
@st.cache_resource
def load_model():
    tokenizer = Tokenizer("Llama-3.2-1B/original/tokenizer.model")
    model = Llama3Model(LLAMA32_CONFIG_1B)
    
    model_path = "model/instruction_fine_tuned_llama32_rlrefined.pth"
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

# --- UI ---
st.title("🧠 RL-Refined LLaMA RL Interface")
user_prompt = st.text_area("Enter your prompt:")

if st.button("Generate Response") and user_prompt.strip():
    with st.spinner("Generating..."):
        sys_prompt = "You are a helpful assistant."
        full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{sys_prompt}<|eot_id|>\n" + \
                      f"<|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot_id|>\n" + \
                      "<|start_header_id|>assistant<|end_header_id|>\n"

        idx = text_to_token_ids(full_prompt, tokenizer).to(device)

        output_ids = generate_v2(
            model=model,
            idx=idx,
            max_new_tokens=llama_generation_parameters["max_new_tokens"],
            context_size=1024,
            temperature=llama_generation_parameters["temperature"],
            top_k=llama_generation_parameters["top_k"],
            eos_id=[128001, 128009],
            frequency_penalty=llama_generation_parameters["frequency_penalty"],
            presence_penalty=llama_generation_parameters["presence_penalty"]
        )

        new_ids = output_ids[:, idx.shape[1]:]
        decoded_response = token_ids_to_text(new_ids, tokenizer)

        st.markdown("### 🤖 Response")
        st.success(decoded_response)
