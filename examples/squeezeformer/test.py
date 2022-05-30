# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from tqdm import tqdm
import argparse
from scipy.special import softmax
import datasets

import tensorflow as tf

from src.configs.config import Config
from src.datasets.asr_dataset import ASRSliceDataset
from src.featurizers.speech_featurizers import TFSpeechFeaturizer
from src.featurizers.text_featurizers import SentencePieceFeaturizer
from src.models.conformer import ConformerCtc
from src.utils import env_util, file_util

logger = env_util.setup_environment()

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

tf.keras.backend.clear_session()

def parse_arguments():
    parser = argparse.ArgumentParser(prog="Conformer Testing")

    parser.add_argument("--config", type=str, default=DEFAULT_YAML, help="The file path of model configuration file")
    parser.add_argument("--mxp", default=False, action="store_true", help="Enable mixed precision")
    parser.add_argument("--device", type=int, default=0, help="Device's id to run test on")
    parser.add_argument("--cpu", default=False, action="store_true", help="Whether to only use cpu")
    parser.add_argument("--saved", type=str, default=None, help="Path to saved model")
    parser.add_argument("--output", type=str, default=None, help="Result filepath")

    # Dataset arguments
    parser.add_argument("--bs", type=int, default=None, help="Test batch size")
    parser.add_argument("--dataset_path", type=str, required=True, help="path to the tsv manifest files")
    parser.add_argument("--dataset", type=str, default="test_other", 
                        choices=["dev_clean", "dev_other", "test_clean", "test_other"], help="Testing dataset")
    parser.add_argument("--input_padding", type=int, default=3700)
    parser.add_argument("--label_padding", type=int, default=530)

    # Architecture arguments
    parser.add_argument("--fixed_arch", default=None, help="force fixed architecture")

    # Decoding arguments
    parser.add_argument("--beam_size", type=int, default=None, help="ctc beam size")

    args = parser.parse_args()
    return args


def parse_fixed_arch(args):
    parsed_arch = args.fixed_arch.split('|')
    i, rep = 0, 1
    fixed_arch = []
    while i < len(parsed_arch):
        if parsed_arch[i].isnumeric():
            rep = int(parsed_arch[i])
        else:
            block = parsed_arch[i].split(',')
            assert len(block) == NUM_LAYERS_IN_BLOCK
            for _ in range(rep):
                fixed_arch.append(block)
            rep = 1
        i += 1
    return fixed_arch

args = parse_arguments()

config = Config(args.config)

NUM_BLOCKS = config.model_config['encoder_num_blocks']
NUM_LAYERS_IN_BLOCK = 4

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.mxp})
env_util.setup_devices([args.device], cpu=args.cpu)

speech_featurizer = TFSpeechFeaturizer(config.speech_config)

logger.info("Use SentencePiece ...")
text_featurizer = SentencePieceFeaturizer(config.decoder_config)

tf.random.set_seed(0)

# Parse fixed architecture
if args.fixed_arch is not None:
    fixed_arch = parse_fixed_arch(args)
    if len(fixed_arch) != NUM_BLOCKS:
        logger.warn(
            f"encoder_num_blocks={config.model_config['encoder_num_blocks']} is " \
            f"different from len(fixed_arch) = {len(fixed_arch)}." \
        )
        logger.warn(f"Changing `encoder_num_blocks` to {len(fixed_arch)}")
        config.model_config['encoder_num_blocks'] = len(fixed_arch)
    logger.info(f"Changing fixed arch: {fixed_arch}")
    config.model_config['encoder_fixed_arch'] = fixed_arch

if args.dataset_path is not None:
    dataset_path = os.path.join(args.dataset_path, f"{args.dataset}.tsv")
    logger.info(f"dataset: {args.dataset} at {dataset_path}")
    config.learning_config.test_dataset_config.data_paths = [dataset_path]
else:
    raise ValueError("specify the manifest file path using --dataset_path")

test_dataset = ASRSliceDataset(
    speech_featurizer=speech_featurizer,
    text_featurizer=text_featurizer,
    input_padding_length=args.input_padding,
    label_padding_length=args.label_padding,
    **vars(config.learning_config.test_dataset_config)
)

conformer = ConformerCtc(
    **config.model_config, 
    vocabulary_size=text_featurizer.num_classes, 
)

conformer.make(speech_featurizer.shape)

if args.saved:
    conformer.load_weights(args.saved, by_name=True)
else:
    logger.warning("Model is initialized randomly, please use --saved to assign checkpoint")
conformer.summary(line_length=100)
conformer.add_featurizers(speech_featurizer, text_featurizer)

batch_size = args.bs or config.learning_config.running_config.batch_size
test_data_loader = test_dataset.create(batch_size)

blank_id = text_featurizer.blank

true_decoded = []
pred_decoded = []
beam_decoded = []

#for batch in enumerate(test_data_loader):
for k, batch in tqdm(enumerate(test_data_loader)):
    labels, labels_len = batch[1]['labels'], batch[1]['labels_length']

    outputs = conformer(batch[0], training=False)
    logits, logits_len = outputs['logits'], outputs['logits_length']
    probs = softmax(logits)

    if args.beam_size is not None:
        beam = tf.nn.ctc_beam_search_decoder(
            tf.transpose(logits, perm=[1, 0, 2]), logits_len, beam_width=args.beam_size, top_paths=1,
        )
        beam = tf.sparse.to_dense(beam[0][0]).numpy()

    for i, (p, l, label, ll) in enumerate(zip(probs, logits_len, labels, labels_len)):
        # p: length x characters
        pred = p[:l].argmax(-1)
        decoded_prediction = []
        previous = blank_id

        # remove the repeting characters and the blanck characters
        for p in pred:
            if (p != previous or previous == blank_id) and p != blank_id:
                decoded_prediction.append(p)
            previous = p

        if len(decoded_prediction) == 0:
            decoded = ""
        else:
            decoded = text_featurizer.iextract([decoded_prediction]).numpy()[0].decode('utf-8')
        pred_decoded.append(decoded)
        label_len = tf.math.reduce_sum(tf.cast(label != 0, tf.int32))
        true_decoded.append(text_featurizer.iextract([label[:label_len]]).numpy()[0].decode('utf-8'))

        if args.beam_size is not None:
            b = beam[i]
            previous = blank_id

            # remove the repeting characters and the blanck characters
            beam_prediction = []
            for p in b:
                if (p != previous or previous == blank_id) and p != blank_id:
                    beam_prediction.append(p)
                previous = p

            if len(beam_prediction) == 0:
                decoded = ""
            else:
                decoded = text_featurizer.iextract([beam_prediction]).numpy()[0].decode('utf-8')
            beam_decoded.append(decoded)

wer_metric = datasets.load_metric("wer")
logger.info(f"Length decoded: {len(true_decoded)}")
logger.info(f"WER: {wer_metric.compute(predictions=pred_decoded, references=true_decoded)}")

if args.beam_size is not None:
    logger.info(f"WER-beam: {wer_metric.compute(predictions=beam_decoded, references=true_decoded)}")


if args.output is not None:
    with file_util.save_file(file_util.preprocess_paths(args.output)) as filepath:
        overwrite = True
        if tf.io.gfile.exists(filepath):
            overwrite = input(f"Overwrite existing result file {filepath} ? (y/n): ").lower() == "y"
        if overwrite:
            logger.info(f"Saving result to {args.output} ...")
            with open(filepath, "w") as openfile:
                openfile.write("PATH\tDURATION\tGROUNDTRUTH\tGREEDY\tBEAMSEARCH\n")
                progbar = tqdm(total=test_dataset.total_steps, unit="batch")
                for i, (groundtruth, greedy) in enumerate(zip(true_decoded, pred_decoded)):
                    openfile.write(f"N/A\tN/A\t{groundtruth}\t{greedy}\tN/A\n")
                    progbar.update(1)
                progbar.close()
