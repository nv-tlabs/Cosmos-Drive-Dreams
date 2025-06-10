# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import click

import attrs
import numpy as np
from tqdm import tqdm
from transformers import T5EncoderModel, T5TokenizerFast
import pickle
from copy import deepcopy
import pandas as pd

t5_path = "/mnt/scratch/cache/imageinaire/google-t5/t5-11b"
tokenizer = T5TokenizerFast.from_pretrained(t5_path)
encoder = T5EncoderModel.from_pretrained(t5_path)
encoder.to("cuda")
encoder.eval()
encoder.requires_grad_(False)

PREFIX_PROMPTS = {
    "pinhole_front": "The video is captured from a camera mounted on a car. The camera is facing forward.",
    "pinhole_front_left": "The video is captured from a camera mounted on a car. The camera is facing forward and slightly to the left.",
    "pinhole_front_right": "The video is captured from a camera mounted on a car. The camera is facing forward and slightly to the right.",
    "pinhole_side_left": "The video is captured from a camera mounted on a car. The camera is facing to the left.",
    "pinhole_side_right": "The video is captured from a camera mounted on a car. The camera is facing to the right.",
}

@attrs.define
class EncodedSample:
    encoded_text: np.ndarray
    length: int
    attn_mask: np.ndarray
    offset_mappings: np.ndarray

    def truncate(self) -> None:
        self.encoded_text = self.encoded_text[0 : self.length].astype(np.float32)
        self.attn_mask = self.attn_mask[0 : self.length].astype(np.int32)
        if self.offset_mappings is not None:
            self.offset_mappings = self.offset_mappings[0 : self.length].astype(np.int32)

_max_length = 512
_output_mapping = True

def _encode_for_batch(
    prompts: list[str],
    truncate: bool = True,
):
    batch_encoding = tokenizer.batch_encode_plus(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=_max_length,
        return_length=True,
        return_offsets_mapping=_output_mapping,
    )

    # We expect all the processing is done in GPU.
    input_ids = batch_encoding.input_ids.cuda()
    attn_mask = batch_encoding.attention_mask.cuda()
    if _output_mapping:
        offsets_mapping = batch_encoding["offset_mapping"]
        offsets_mapping = offsets_mapping.cpu().numpy()
    else:
        offsets_mapping = None

    outputs = encoder(input_ids=input_ids, attention_mask=attn_mask)  # type: ignore
    encoded_text = outputs.last_hidden_state

    lengths = attn_mask.sum(dim=1).cpu()  # batch_encoding["lengths"] is not valid for T5TokenizerFast
    for batch_id in range(encoded_text.shape[0]):
        encoded_text[batch_id][lengths[batch_id] :] = 0

    encoded_text = encoded_text.cpu().numpy()
    attn_mask = attn_mask.cpu().numpy()

    encoded_text = encoded_text[:, :_max_length]
    attn_mask = attn_mask[:, :_max_length]

    out = []  # type: list[EncodedSample]
    for idx in range(encoded_text.shape[0]):
        if _output_mapping:
            offsets = offsets_mapping[idx]
        else:
            offsets = None

        out.append(EncodedSample(encoded_text[idx].astype(np.float32), lengths[idx], attn_mask[idx], offsets))
    if truncate:
        for x in out:
            x.truncate()
    return out

@click.command()
@click.option(
    "--caption_file",
    type=str,
    default='./assets/waymo_caption.csv',
    help="Path to the folder containing caption files.",
)
@click.option("--data_root", type=str, default="waymo_mv", help="Path to data root directory")

def main(caption_file, data_root):
    """
    Main function to create T5 embeddings from caption files.
    """
    save_embd_folder = os.path.join(data_root, "t5_xxl")
    save_prefix_emb_folder = os.path.join(data_root, "cache")
    videos_folder = os.path.join(data_root, "videos", "pinhole_front")
    os.makedirs(os.path.join(data_root, "t5_xxl", "pinhole_front"), exist_ok=True)
    if save_prefix_emb_folder:
        os.makedirs(save_prefix_emb_folder, exist_ok=True)
        for view_name, prefix_prompt in PREFIX_PROMPTS.items():
            t5_xxl_filename = os.path.join(save_prefix_emb_folder, f"prefix_{view_name}.pkl")
            os.makedirs(os.path.dirname(t5_xxl_filename), exist_ok=True)
            if os.path.exists(t5_xxl_filename):
                # Skip if the file already exists
                continue

            # Compute T5 embeddings
            encoded_text = _encode_for_batch([prefix_prompt])

            # Save T5 embeddings as pickle file
            with open(t5_xxl_filename, "wb") as fp:
                pickle.dump(encoded_text[0].encoded_text, fp)

    # Load the caption files and create embeddings
    df = pd.read_csv(caption_file, header=None)
    key_list = df.iloc[0].tolist()
    caption_list = df.iloc[1].tolist()
    if videos_folder:
        video_paths = [os.path.join(videos_folder, f) for f in os.listdir(videos_folder) if f.endswith(".mp4")]
        video_id_to_name = {}
        for vpath in video_paths:
            vname = os.path.basename(vpath).split(".")[0]
            if vname.endswith("_0"):
                vname = vname[:-2]
            video_id_to_name[vname] = vpath
    prompts = []
    all_keys = []
    caption = {}
    for key, cap in zip(key_list, caption_list):
        # check if already processed
        if os.path.exists(os.path.join(save_embd_folder, 'pinhole_front', key + ".pkl")):
            continue
        if videos_folder:
            if not key in video_id_to_name:
                continue
        all_keys.append(key)
        prompts.append(cap)
        caption[all_keys[-1]] = prompts[-1]

    batch_size = 100
    n_batch = len(prompts) // batch_size + 1

    for i in tqdm(range(n_batch)):
        prompt = prompts[i * batch_size : min((i + 1) * batch_size, len(prompts))]
        out = _encode_for_batch(prompt)
        for j in range(len(out)):
            emb = out[j]
            k = all_keys[i * batch_size + j]
            with  open(os.path.join(save_embd_folder, 'pinhole_front', str(k) + ".pkl"), "wb") as fp:
                pickle.dump([emb.encoded_text], fp)


if __name__ == "__main__":
    main()