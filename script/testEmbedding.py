from pathlib import Path
cwd = Path.cwd()
file_path = cwd.joinpath('data','NYT','combined.embed')

def test1():
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(lines[0])
    lines = lines[1:]
    words = {}
    for idx,line in enumerate(lines):
        t = line.strip().split(' ')
        word, _ = t[0].split('||')
        if word in words:
            print(word)
        words[word] = idx

def test2():
    filepath = cwd.joinpath('data','NYT','train.set')
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        # print(line)
        s = line.split('{')[1][:-2]
        words = [eval(i) for i in s.split(',')]
        print(words)

if __name__ == '__main__':
    test2()
