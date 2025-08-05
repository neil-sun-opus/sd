from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
import torch

# Load model
model = EaModel.from_pretrained(
    base_model_path="/home/neil/sd/models/vicuna-13b-v1.3",
    ea_model_path="/home/neil/sd/EAGLE/models/eagle3-vicuna-13b-v1.3",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Prepare input
user_message = "Explain quantum computing in simple terms."
conv = get_conversation_template("vicuna")  # or "llama2" or "llama3" based on your model
conv.append_message(conv.roles[0], user_message)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

# Tokenize input
input_ids = model.tokenizer([prompt]).input_ids
input_ids = torch.as_tensor(input_ids).cuda()

# Generate text
output_ids = model.eagenerate(
    input_ids,
    temperature=0,
    max_new_tokens=512,
    log = True
)

# Decode output
output = model.tokenizer.decode(output_ids[0])
import pdb; pdb.set_trace()
print(output)