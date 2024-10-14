## Inference python scripts
import os
import sys
import argparse
import logging
import torch

sys.path.append(os.getcwd())

from typing import Sequence
from typing import Union
from typing import Tuple
from typing import Optional

from funcodec.bin.text2audio_inference import Text2Audio, save_audio
from funcodec.tasks.text2audio_generation import Text2AudioGenTask
from utils import setup_logger, update_args, setup_seed


def inference_func(
    output_dir: Optional[str] = None,
    batch_size: int = 1,
    dtype: str = "float32",
    device: str = "cuda",
    logging: logging.Logger = logging.getLogger(),
    num_workers: int = 0,
    key_file: Optional[str] = None,
    config_file: Optional[str] = "config.yaml",
    model_file: Optional[str] = "model.pth",
    model_tag: Optional[str] = None,
    allow_variable_data_keys: bool = True,
    streaming: bool = False,
    **kwargs,
):
    """
    copied from funcodec.bin.text2audio_inference.inference_func
    """

    # 2. Build model
    model_kwargs = dict(
        config_file=config_file,
        model_file=model_file,
        device=device,
        dtype=dtype,
        streaming=streaming,
        **kwargs,
    )
    my_model = Text2Audio.from_pretrained(
        model_tag=model_tag,
        **model_kwargs,
    )
    my_model.model.eval()

    def _forward(
        data_path_and_name_and_type: Sequence[Tuple[str, str, str]] = None,
        raw_inputs: Union[Tuple[str], Tuple[str, str, str]] = None,
        output_dir_v2: Optional[str] = None,
        param_dict: Optional[dict] = None,
    ):
        logging.info("param_dict: {}".format(param_dict))
        if data_path_and_name_and_type is None and raw_inputs is not None:
            # add additional parenthesis to keep the same data format as streaming_iterator
            logging.info("infering on one audio data")
            data_dict = dict(text=[raw_inputs[0]])
            if len(raw_inputs) == 3:
                data_dict["prompt_text"] = [raw_inputs[1]]
                if isinstance(raw_inputs[2], str):
                    data_dict["prompt_audio"] = [
                        librosa.load(
                            raw_inputs[2],
                            sr=my_model.codec_model.model.quantizer.sampling_rate,
                            mono=True,
                            dtype=np.float32,
                        )[0][np.newaxis, :]
                    ]
                else:
                    data_dict["prompt_audio"] = [raw_inputs[2].squeeze()[None, :]]
            loader = [(["utt1"], data_dict)]
        else:
            loader = Text2AudioGenTask.build_streaming_iterator(
                data_path_and_name_and_type,
                dtype=dtype,
                batch_size=batch_size,
                key_file=key_file,
                num_workers=num_workers,
                preprocess_fn=None,
                collate_fn=Text2AudioGenTask.build_collate_fn(
                    my_model.model_args, False, raw_sequence=("text", "prompt_text")
                ),
                allow_variable_data_keys=allow_variable_data_keys,
                inference=True,
            )

        output_path = output_dir_v2 if output_dir_v2 is not None else output_dir
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        result_list = []

        for keys, data in loader:
            key = keys[0]
            logging.info(f"generating {key}")
            model_inputs = [data["text"][0]]
            for input_key in ["prompt_text", "prompt_audio"]:
                if input_key in data:
                    model_inputs.append(data[input_key][0])

            ret_val, _ = my_model(*model_inputs)
            item = {"key": key, "value": ret_val}
            if output_path is not None:
                for suffix, wave in ret_val.items():
                    file_name = key.replace(".wav", "") + "_" + suffix + ".wav"
                    save_path = os.path.join(output_path, file_name)
                    save_audio(
                        wave[0],
                        save_path,
                        rescale=True,
                        sample_rate=my_model.codec_model.model.quantizer.sampling_rate,
                    )
            else:
                result_list.append(item)

        return result_list

    return _forward


def main(args: argparse.Namespace):

    ## TODO: delete this code and write the code in the train.py
    setup_seed(args.seed, 0)
    ##
    logger = setup_logger(args, rank=0, out=False)

    ckpt = torch.load(args.model_file, map_location="cuda")
    torch.save(ckpt["model_state_dict"], "./.temp.pth")
    args.model_file = "./.temp.pth"
    logger.info("model ckpt is done, please change this ASAP!")
    args.logging = logger
    logger.info(args)
    forward = inference_func(**vars(args))
    # forward(data_path_and_name_and_type= args.)
    forward(args.data_path_and_name_and_type)

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--default_config", type=str)
    parser.add_argument("--model_file", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--raw_inputs", nargs="*", default=None, type=str)
    parser.add_argument("--tokenize_to_phone", action="store_true")
    args = parser.parse_args()
    update_args(args, args.default_config)
    main(args)
    pass
