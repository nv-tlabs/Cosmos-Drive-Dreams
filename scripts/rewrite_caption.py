import os
from tqdm import tqdm
from itertools import cycle
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import click
import json

CHUNK_LEN = 25
USE_RAY = False
AUG_COLOR = True
C0_ONLY = False
REWRITE_TYPES = ["Golden hour", "Morning", "Night", "Rainy", "Snowy", "Sunny", "Foggy"]

if USE_RAY:
    import ray
    decorator = ray.remote(num_gpus=1)
else:
    def decorator(func):
        return func

class QwenChatbot:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.history = []

    def initialize(self, messages):
        self.history.extend(messages)

    def generate_response(self, user_input):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response

def rewrite_caption(caption, model, tokenizer, rewrite_type):
    bot = QwenChatbot(tokenizer, model)

    system_prompt = """
    You are a prompt optimization specialist. Your task is to rewrite user-provided input prompts into high-quality English descriptions by modifying specific temporal or environmental details, while preserving the core content and actions of the original scene. \n
    There are two types of rewrites: \n
    1. Time of Day: Change the time setting in the caption, including Golden hour (with long shadows), Morning, and Night. \n
    2. Environment/Weather: Change the weather condition in the caption, including Rainy, Snowy, Sunny, Foggy. \n

    Requirements:
	- Keep the scene and actions the same (e.g., a car driving down a highway should still be a car driving down a highway).
	- Change only the details related to time or environment as instructed.
	- Ensure the rewrite matches the new condition (e.g., no mention of sun glare in a foggy or snowy version).
    """

    user_prompt = f"""
    Rewrite the following caption to include specific environmental or temporal details. \n
    Original Caption: {caption} \n
    Rewrite Type: {rewrite_type} \n
    Please provide a detailed and high-quality rewrite that maintains the core content of the scene. Format your response by having the rewritten caption following 'New caption:' /no_think
    """

    messages = [
        {"role": "system", "content": system_prompt},
        #{"role": "user", "content": user_prompt}
    ]
    bot.initialize(messages)
    response = bot.generate_response(user_prompt)
    for i in range(3):
        new_prompt = response.split("New caption:")
        if len(new_prompt) == 1:
            response = bot.generate_response("make sure your response starts with 'New caption:' /no_think")
            new_prompt = response.split("New caption:")
        else:
            new_prompt = new_prompt[-1]
            break
    if AUG_COLOR:
        _ = bot.generate_response("List all occurrences in the text where a vehicle is described with a color. If there is no specific colors mentioned in the prompt, do not add anything /no_think")
        response_3 = bot.generate_response(f"""
        For every occurence mentioned above, re-write the original text with a different color. For example, red sedan becomes white sedan. Include only the text in the response. If there is no specific colors mentioned in the prompt, return the original text.\n
        Original text: {new_prompt} /no_think
        """)
        response_3 = response_3.removeprefix("<think>\n\n</think>\n\n")
        return response_3

    return new_prompt


@decorator
def process_video(txt_file_path_list, output_json_path_list):
    # if txt_file_path is list
    if not isinstance(txt_file_path_list, list):
        txt_file_path_list = [txt_file_path_list]
        output_json_path_list = [output_json_path_list]

    model_name = "Qwen/Qwen3-14B"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for txt_file_path, output_json_path in zip(txt_file_path_list, output_json_path_list):
        with open(txt_file_path, 'r') as f:
            input_caption = f.read()

        output_captions = {
            "Original": input_caption,
        }
        for rewrite_type in REWRITE_TYPES:
            # Generate a rewritten caption
            output_caption = rewrite_caption(input_caption, model, tokenizer, rewrite_type)
            output_caption = output_caption.replace("Rewritten Caption: ", "")
            output_caption = output_caption.replace("\n", " ").replace("  ", " ")
            print(f"[Caption] {input_caption}")
            print(f"[Rewritten Caption] {output_caption}")
            output_captions[rewrite_type] = output_caption

        # Save the output captions to a JSON file
        with open(output_json_path, 'w') as f:
            json.dump(output_captions, f, indent=4)

@click.command()
@click.option("--input_dir", '-i', type=str, help="the root folder of the input data")
@click.option("--output_dir", '-o', type=str, help="the root folder of the output data")
def main(input_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get the list of all .txt files
    if C0_ONLY:
        txt_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('_0.txt')]
    else:
        txt_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.txt')]

    # Filter out txt files that already have a corresponding output text file
    txt_files_to_process = []

    cur_chunk_size = 0
    cur_txt_files = []
    cur_output_json_files = []

    for txt_file in tqdm(txt_files):
        output_json_path = os.path.join(
            output_dir, os.path.basename(txt_file).replace('.txt', '.json')
        )
        if not os.path.exists(output_json_path):
            cur_txt_files.append(txt_file)
            cur_output_json_files.append(output_json_path)
            cur_chunk_size += 1

        if cur_chunk_size == CHUNK_LEN:
            txt_files_to_process.append((cur_txt_files, cur_output_json_files))
            cur_txt_files = []
            cur_output_json_files = []
            cur_chunk_size = 0

    # Process the remaining files
    if cur_chunk_size > 0:
        txt_files_to_process.append((cur_txt_files, cur_output_json_files))

    if len(txt_files_to_process) == 0:
        print("All txt files have been processed.")
        return

    if USE_RAY:
        # Initialize Ray

        ray.init(address="auto")

        # Distribute the tasks among the actors
        futures = []
        for txt_file, output_json_path in txt_files_to_process:
            future = process_video.remote(txt_file, output_json_path)
            futures.append(future)

        # Monitor progress using tqdm
        progress_bar = tqdm(total=len(futures), desc="Processing videos")
        while len(futures):
            done_id, futures = ray.wait(futures)
            progress_bar.update(len(done_id))
            for obj_ref in done_id:
                try:
                    ray.get(obj_ref)
                except Exception as e:
                    print(f"Exception in processing video: {e}")
        progress_bar.close()

        # Shutdown Ray
        ray.shutdown()
    else:
        for txt_file, output_json_path in txt_files_to_process:
            process_video(txt_file, output_json_path)

if __name__ == "__main__":
    main()