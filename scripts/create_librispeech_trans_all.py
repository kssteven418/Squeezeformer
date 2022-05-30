import os
import csv
import subprocess
import argparse

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'test-only'])
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    return args

args = arg_parse()

for n in ['dev', 'test']:
    for m in ['clean', 'other']:
        outname = f'{n}_{m}.tsv'
        inname = f'{n}-{m}'
        print(f'processing {inname}')
        subprocess_args = [
            'python', 'create_librispeech_trans.py', os.path.join(args.output_dir, outname),
            '--dir', os.path.join(args.dataset_dir, inname)
        ]

        subprocess.call(subprocess_args)

if args.mode == 'all':
    train_set_names = [
        ('train-clean-100', 'train_clean_100.tsv'), 
        ('train-clean-360', 'train_clean_360.tsv'), 
        ('train-other-500', 'train_other_500.tsv'), 
    ]

    for inname, outname in train_set_names:
        print(f'processing {inname}')
        subprocess_args = [
            'python', 'create_librispeech_trans.py', os.path.join(args.output_dir, outname),
            '--dir', os.path.join(args.dataset_dir, inname)
        ]

        subprocess.call(subprocess_args)

    lines = ["PATH\tDURATION\tTRANSCRIPT\n"]

    tsv_names = [x[-1] for x in train_set_names]
    for tsv_name in tsv_names:
        infile = os.path.join(args.output_dir, tsv_name)


        with open(infile) as file:
            tsv_file = csv.reader(file, delimiter="\t")
            for i, line in enumerate(tsv_file):
                if i == 0: continue
                audio_file, duration, text = line
                lines.append(f"{audio_file}\t{duration}\t{text}\n")

    output_file = os.path.join(args.output_dir, 'train_all.tsv')
    with open(output_file, "w", encoding="utf-8") as out:
        for line in lines:
            out.write(line)
