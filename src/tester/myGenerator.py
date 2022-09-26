import sys
import os
import re
import json
import shutil
import subprocess as sp
from pathlib import Path

TESTER_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
DATA_DIR = TESTER_DIR + '../../data/data/'
TESTER_DIR_PATH = Path(TESTER_DIR)
sys.path.append(TESTER_DIR + '../../data/data/')
sys.path.append(TESTER_DIR + '../validation/')
sys.path.append(TESTER_DIR + '../dataloader/')

import tokenization

from prepare_testing_data import prepare_cure_input
from generator import generate_gpt_conut
from generator import generate_gpt_fconv
from rerank import cure_rerank
from rerank import read_defects4j_meta
from validate_defects4j import get_strings_numbers
from validate_defects4j import insert_fix_defects4j

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

def getMutRelativeSourcePath(projPath: Path, mutId: str):
    mutantDir = projPath / 'mutants' / mutId
    mutantSourcePathStr = sp.check_output("find . -name '*.java'", shell=True, text=True, cwd=str(mutantDir)).strip()
    return mutantSourcePathStr

def getMutReplaceRelativePath(projPath: Path, mutId: str):
    mutantDir = projPath / 'mutants' / mutId
    mutantSourcePathStr = sp.check_output("find . -name '*.java'", shell=True, text=True, cwd=str(mutantDir)).strip()
    projSrcRelativePath = sp.check_output("defects4j export -p dir.src.classes", shell=True, text=True, cwd=str(projPath)).strip()
    return projSrcRelativePath + '/' + mutantSourcePathStr


def prepareMutInput(projPath: Path, projName: str):
    for mutId in getMutIds(projPath):
        print('****** Preparing {} Mutant-{} ******'.format(projName, mutId))
        mutDataDir = TESTER_DIR_PATH / (projName + '-mutants') / mutId
        if fileExistsAndNotEmpty(mutDataDir / 'input.txt'):
            continue
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

def fileExistsAndNotEmpty(p: Path):
    return p.exists() and p.stat().st_size > 0

def generatePatches(projPath: Path, projName: str):
    for mutId in getMutIds(projPath):
        print('****** Patching {} Mutant-{} ******'.format(projName, mutId))
        try:
            mutDataDir = TESTER_DIR_PATH / (projName + '-mutants') / mutId
            mutDataDir.resolve()
            vocab_file = TESTER_DIR + '../../data/vocabulary/vocabulary.txt'
            input_file = str(mutDataDir / 'input.txt')
            identifier_txt_file = str(mutDataDir / 'identifier.txt')
            identifier_token_file = str(mutDataDir / 'identifier.tokens')
            assert mutDataDir.exists() and os.path.exists(vocab_file) and os.path.exists(input_file) and os.path.exists(identifier_txt_file) and os.path.exists(identifier_token_file)

            beam_size = 1000
            os.environ['CUDA_VISIBLE_DEVICES'] = "1"

            model_file = TESTER_DIR + '../../data/models/gpt_conut_1.pt'
            output_file = str(mutDataDir / 'gpt_conut_1.txt')
            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                generate_gpt_conut(vocab_file, model_file, input_file, identifier_txt_file, identifier_token_file, output_file, beam_size)

            model_file = TESTER_DIR + '../../data/models/gpt_fconv_1.pt'
            output_file = str(mutDataDir / 'gpt_fconv_1.txt')
            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                generate_gpt_fconv(vocab_file, model_file, input_file, identifier_txt_file, identifier_token_file, output_file, beam_size)
        except:
            import traceback
            traceback.print_exc()
            print('[ERROR] Failed to generate patch for {} mutant-{}'.format(projName, mutId))

def combineOutputs(projPath: Path, projName: str):
    outputFile1 = TESTER_DIR_PATH / (projName + '-mutants') / 'gpt_conut_1.txt'
    outputFile2 = TESTER_DIR_PATH / (projName + '-mutants') / 'gpt_fconv_1.txt'
    outputFile1.unlink(missing_ok=True)
    outputFile2.unlink(missing_ok=True)
    cnt = 0

    with outputFile1.open(mode='a') as o1:
        with outputFile2.open(mode='a') as o2:
            for mutId in getMutIds(projPath):
                print('****** Combining {} Mutant-{} ******'.format(projName, mutId))
                mutDataDir = TESTER_DIR_PATH / (projName + '-mutants') / mutId
                mutDataDir.resolve()

                file1 = mutDataDir / 'gpt_conut_1.txt'
                file2 = mutDataDir / 'gpt_fconv_1.txt'

                if not file1.exists() or not file2.exists() or file1.stat().st_size == 0 or file2.stat().st_size == 0:
                    print("Skipping Mutant-{}".format(mutId))
                    continue

                with file1.open() as f1:
                    for line in f1:
                        line = re.sub('(\w)-0', r'\1-' + str(cnt), line, count=1)
                        o1.write(line)
                with file2.open() as f2:
                    for line in f2:
                        line = re.sub('(\w)-0', r'\1-' + str(cnt), line, count=1)
                        o2.write(line)
                cnt += 1

def generateMeta(projPath: Path, projName: str):
    file = TESTER_DIR_PATH / (projName + '-mutants') / 'meta.txt'
    with file.open('a') as f:
        for mutId in getMutIds(projPath):
            lineNum = getMutLineNum(projPath, mutId)
            res = 'Mutant\t{}\t{}\t{}\t{}\n'.format(mutId, getMutReplaceRelativePath(projPath, mutId), lineNum, lineNum)
            f.write(res)

def rerank(projPath: Path, projName: str):
    mutDataDir = TESTER_DIR_PATH / (projName + '-mutants')
    mutDataDir.resolve()
    metaFile = TESTER_DIR_PATH / (projName + '-mutants') / 'meta.txt'
    metaFile.resolve()
    if not metaFile.exists():
        generateMeta(projPath, projName)
    quixbugs_meta = read_defects4j_meta(str(metaFile))
    hypo_path_list = [str(mutDataDir / 'gpt_conut_1.txt')] + [str(mutDataDir / 'gpt_fconv_1.txt')]
    output_path = str(mutDataDir / 'reranked_patches.json')
    cure_rerank(quixbugs_meta, hypo_path_list, output_path)

def compilePatches(projPath: Path, projName: str):
    tmp_dir = str(projPath) + '/'
    projClassDirPath = sp.check_output("defects4j export -p dir.bin.classes", shell=True, text=True, cwd=str(projPath)).strip()
    mutDataDir = TESTER_DIR_PATH / (projName + '-mutants')
    mutDataDir.resolve()
    reranked_result_path = str(mutDataDir / 'reranked_patches.json')

    patchId = 0

    reranked_result = json.load(open(reranked_result_path, 'r'))
    for key in reranked_result:
        
        proj, bug_id, path, start_loc, end_loc = key.split('-')

        print('===== Mutant-{} ====='.format(bug_id))
        
        # bug_start_time = time.time()
        i = 0
        for tokenized_patch in reranked_result[key]['patches']:
            print('***** Patch-{} *****'.format(i))

            i += 1
            # validate 5 hours for each bug at most
            # if time.time() - bug_start_time > 5 * 3600:
                # break
            # validate 5000 patches for each bug at most
            # if len(validated_result[key]['patches']) >= 5000:
            #     break

            score = tokenized_patch['score']
            tokenized_patch = tokenized_patch['patch']
            print("tokenized_patch: '{}'".format(str(tokenized_patch)))


            strings, numbers = get_strings_numbers(tmp_dir + path, (int(start_loc) + int(end_loc)) // 2)
            strings = [item[0] for item in strings][:5]
            numbers = [item[0] for item in numbers][:5]
            print('strings: ' + str(strings))
            print('numbers: ' + str(numbers))
            # one tokenized patch may be reconstructed to multiple source-code patches
            reconstructed_patches = tokenization.token2statement(tokenized_patch.split(' '), numbers, strings)
            print('reconstructed_patches: ' + str(reconstructed_patches))
            # validate most 5 source-code patches come from the same tokenized patch
            for patch in reconstructed_patches[:5]:
                patch = patch.strip()

                patched_file = insert_fix_defects4j(path, int(start_loc), int(end_loc), patch, tmp_dir)
                sourcePath = projPath / path

                FNULL = open(os.devnull, 'w')
                process = sp.Popen('defects4j compile', shell=True, text=True, stdout=FNULL, stderr=FNULL)
                process.communicate()
                ret_code = process.poll()
                if ret_code == 0:
                    print('Compile Succeeded! PatchId: {}'.format(patchId))
                    relativeSourcePath = getMutRelativeSourcePath(projPath, bug_id)
                    targetPatchSourcePathStr = (projName + '-mutants') + '/patches-pool/{}/{}'.format(patchId, relativeSourcePath)  # xxx.java
                    Path(targetPatchSourcePathStr).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(sourcePath, targetPatchSourcePathStr)
                    shutil.copyfile(projClassDirPath + '/' + relativeSourcePath[:-5]+".class", targetPatchSourcePathStr[:-5] + ".class")
                    patchId += 1
                elif ret_code != 0:
                    print('Compile Failed!')
                    # relativeSourcePath = getMutRelativeSourcePath(projPath, bug_id)
                    # targetPatchSourcePathStr = (projName + '-mutants') + '/patches-pool/{}/{}'.format(patchId, relativeSourcePath)  # xxx.java
                    # Path(targetPatchSourcePathStr).parent.mkdir(parents=True, exist_ok=True)
                    # shutil.copyfile(sourcePath, targetPatchSourcePathStr)
                    # patchId += 1

                shutil.copyfile(patched_file, patched_file.replace('.bak', ''))

def err(msg: str):
    print(msg)

def genPatchedClasses(projPath: Path, projName: str):
    prepareMutInput(projPath, projName)
    generatePatches(projPath, projName)
    combineOutputs(projPath, projName)
    generateMeta(projPath, projName)
    rerank(projPath, projName)
    compilePatches(projPath, projName)

if __name__ == "__main__":
    path = Path('/home/yicheng/check-apr/dataset/chart-1f')
    genPatchedClasses(path, 'Chart-1f')

    # path = Path('/home/yicheng/check-apr/dataset/lang-1f')
    # genPatchedClasses(path, 'Lang-1f')
