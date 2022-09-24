import sys
import os
import re
import subprocess as sp
from pathlib import Path

TESTER_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
DATA_DIR = TESTER_DIR + '../../data/data/'
TESTER_DIR_PATH = Path(TESTER_DIR)
sys.path.append(TESTER_DIR + '../../data/data/')

from prepare_testing_data import prepare_cure_input
from generator import generate_gpt_conut
from generator import generate_gpt_fconv

def getMutIds(projPath: Path):
    res = []
    killCsv = projPath / 'kill.csv'
    assert killCsv.exists()
    with killCsv.open() as csv:
        for line in csv:
            if ',FAIL' in line or ',EXC' in line:
                res.append(line.split(',')[0])
    return res

def getMutLineNum(projPath: Path, mutId: str):
    mutLog = projPath / 'mutants.log'
    assert mutLog.exists()
    with mutLog.open() as log:
        for line in log:
            if mutId + ':' in line:
                m = re.match(r'.+:(\d+):[^:]+\n', line)
                assert m is not None
                return int(m[1])

def getMutSourcePath(projPath: Path, mutId: str):
    mutantDir = projPath / 'mutants' / mutId
    mutantSourcePathStr = sp.check_output("find {} -name '*.java'".format(mutantDir.resolve()), shell=True, text=True)
    return mutantSourcePathStr.strip()

def prepareMutInput(projPath: Path, projName: str):
    for mutId in getMutIds(projPath):
        print('****** Preparing {} Mutant-{} ******'.format(projName, mutId))
        mutDataDir = TESTER_DIR_PATH / (projName + '-mutants')
        mutDataDir.mkdir(parents=True, exist_ok=True)
        mutDataDir.resolve()
        buggyLineNum = getMutLineNum(projPath, mutId)
        prepare_cure_input(
            buggy_file=getMutSourcePath(projPath, mutId),
            start_line=buggyLineNum, 
            end_line=buggyLineNum+1,
            java_class_path=DATA_DIR + 'java_class.json',
            java_keyword_path=DATA_DIR + 'java_keyword.json',
            tmp_dir='/tmp/',
            output_dir=str(mutDataDir)
        )

def generatePatches(projPath: Path, projName: str):
    mutDataDir = TESTER_DIR_PATH / (projName + '-mutants')
    mutDataDir.resolve()
    vocab_file = TESTER_DIR + '../../data/vocabulary/vocabulary.txt'
    input_file = str(mutDataDir / 'input.txt')
    identifier_txt_file = str(mutDataDir / 'identifier.txt')
    identifier_token_file = str(mutDataDir / 'identifier.tokens')
    assert mutDataDir.exists() and os.path.exists(vocab_file) and os.path.exists(input_file) and os.path.exists(identifier_txt_file) and os.path.exists(identifier_token_file)

    beam_size = 100
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    model_file = TESTER_DIR + '../../data/models/gpt_conut_1.pt'
    output_file = str(mutDataDir / 'gpt_conut_1.txt')
    generate_gpt_conut(vocab_file, model_file, input_file, identifier_txt_file, identifier_token_file, output_file, beam_size)

    model_file = TESTER_DIR + '../../data/models/gpt_fconv_1.pt'
    output_file = str(mutDataDir / 'gpt_fconv_1.txt')
    generate_gpt_fconv(vocab_file, model_file, input_file, identifier_txt_file, identifier_token_file, output_file, beam_size)

def err(msg: str):
    print(msg)

if __name__ == "__main__":
    path = Path('/home/yicheng/check-apr/dataset/chart-1f')
    # print(getMutIds('/home/yicheng/check-apr/dataset/chart-1f'))
    # print(getMutLineNum(path, '44'))
    # print(getMutSourcePath(path, '44'))
    # prepareMutInput(path, 'Chart-1f')
    generatePatches(path, 'Chart-1f')
    
