from pathlib import Path
cwd = Path.cwd()
import random

def split(dir_path:Path):
    with open(dir_path.joinpath('train-cold.set'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
    sub_l = int(len(lines) * 0.2)
    train = lines[sub_l:]
    dev = lines[:sub_l]
    with open(dir_path.joinpath('train.set'), 'w', encoding='utf-8') as f:
        for i in train:
            f.write(i)
    with open(dir_path.joinpath('dev.set'), 'w', encoding='utf-8') as f:
        for i in dev:
            f.write(i)

if __name__ == '__main__':
    NYT = cwd.joinpath('data', 'NYT')
    PubMed = cwd.joinpath('data', 'PubMed')
    Wiki = cwd.joinpath('data', 'Wiki')
    split(NYT)
    split(PubMed)
    split(Wiki)