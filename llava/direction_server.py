import socket
import json
import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import io
import traceback

print("Loading model...")
model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    offload_buffers=True,
)
processor = LlavaNextProcessor.from_pretrained(model_id)
tokenizer = processor.tokenizer

token_A = tokenizer.encode("A", add_special_tokens=False)[0]
token_B = tokenizer.encode("B", add_special_tokens=False)[0]
token_C = tokenizer.encode("C", add_special_tokens=False)[0]
token_D = tokenizer.encode("D", add_special_tokens=False)[0]

print("Model loaded!")

def process_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    conversation = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Which direction (A, B, C, or D) should I explore to find a bed? Answer with ONLY a single letter: A, B, C, or D. Do not explain."},
        ],
    }]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    next_token_logits = outputs.logits[0, -1, :]
    logit_A = next_token_logits[token_A].item()
    logit_B = next_token_logits[token_B].item()
    logit_C = next_token_logits[token_C].item()
    logit_D = next_token_logits[token_D].item()
    logits_tensor = torch.tensor([logit_A, logit_B, logit_C, logit_D])
    probs = torch.softmax(logits_tensor, dim=0)
    return {
        'probabilities': [probs[0].item(), probs[1].item(), probs[2].item(), probs[3].item()],
        'logits': [logit_A, logit_B, logit_C, logit_D],
        'best_choice': ['A', 'B', 'C', 'D'][probs.argmax()],
        'confidence': probs.max().item()
    }

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('0.0.0.0', 5555))
server.listen(1)
print("✅ Server ready on 0.0.0.0:5555\n")

while True:
    conn, addr = server.accept()
    print(f"🔗 Connected by {addr}")
    try:
        while True:
            size_data = conn.recv(8)
            if not size_data: 
                break
            
            image_size = int.from_bytes(size_data, 'big')
            print(f"📥 Receiving image: {image_size} bytes")
            
            image_data = b''
            while len(image_data) < image_size:
                chunk = conn.recv(min(image_size - len(image_data), 8192))
                if not chunk: 
                    break
                image_data += chunk
            
            if len(image_data) != image_size:
                print(f"❌ Incomplete image: got {len(image_data)}/{image_size} bytes")
                break
            
            print("🔄 Processing image...")
            result = process_image(image_data)
            print(f"✅ Processed! Best: {result['best_choice']}")
            
            result_json = json.dumps(result).encode('utf-8')
            conn.sendall(len(result_json).to_bytes(8, 'big'))
            conn.sendall(result_json)
            print(f"📤 Result sent!\n")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
    finally:
        conn.close()
        print("Connection closed\n")
