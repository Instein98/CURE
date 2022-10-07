import sys
import os
import re
import json
import shutil
import subprocess as sp
from pathlib import Path

from myGenerator import prepareMutInputAllIn
from myGenerator import generatePatchesAllIn
from myGenerator import redirectOutErrToLogsAllin
from myGenerator import generateMeta
from myGenerator import rerank
from myGenerator import recoverOutErr
from myGenerator import getMutFixedLine

TESTER_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
sys.path.append(TESTER_DIR + '../validation/')
from validate_defects4j import get_strings_numbers
sys.path.append(TESTER_DIR + '../dataloader/')
import tokenization

TESTER_DIR_PATH = Path(TESTER_DIR)
mutResultDirPath = TESTER_DIR_PATH / 'mutResults'
mutResultDirPath.resolve()
d4jProjDirPath = Path(TESTER_DIR + '../../../d4jProj')
d4jProjDirPath.resolve()
d4jProjPaths = []
for dir in d4jProjDirPath.iterdir():
    if dir.is_dir():
        d4jProjPaths.append(dir)
# print(d4jProjPaths)

def getSampledMutIdList(projPath: Path):
    res = []
    txt = projPath / 'sampledMutIds.txt'
    assert txt.exists()
    with txt.open() as f:
        for line in f:
            if len(line.strip()) > 0:
                res.append(line.strip())
    return res

def main():
    for projPath in d4jProjPaths:
        projName = projPath.stem

        if "jacksonxml" not in projName:
            continue

        print('=' * 10 + " Start {} ".format(projName) + '=' * 10)
        mutIds=getSampledMutIdList(projPath)
        
        mutDataDir = mutResultDirPath / (projName + '-mutants-allin')
        if (mutDataDir / 'reranked_patches.json').exists():
            print('reranked_patches.json exists in {}'.format(projName))
            continue

        redirectOutErrToLogsAllin(projName)
        try:
            prepareMutInputAllIn(projPath, projName, mutIds=mutIds)
            generatePatchesAllIn(projName)
            generateMeta(projPath, projName, allin=True, mutIds=mutIds)
            rerank(projPath, projName, allin=True)
        except:
            import traceback
            traceback.print_exc()
        recoverOutErr()

def checkCorrectFixes():
    for dir in mutResultDirPath.iterdir():
        if dir.is_dir():
            resultJson = dir / 'reranked_patches.json'
            if resultJson.exists():
                print('='*10 + dir.stem + '='*10)
                projName = dir.stem.split('-')[0]
                projPathStem = re.match(r'(\w+?-\d+f).*', dir.stem)[1]
                projPath = d4jProjDirPath / projPathStem
                print(str(projPath))
                print(projPath.exists())
                # recoverPatches(projPath, resultJson, dir / 'patches', projName)
                # patchDict = json.load()

def recoverPatches(projPath: Path, rankedJsonPath: Path, outputDir: Path, projName: str):
    outputDir.mkdir(exist_ok=True, parents=True)
    tmp_dir = str(projPath) + '/'
    reranked_result_path = str(rankedJsonPath)

    with rankedJsonPath.open() as f:
        reranked_result = json.load(f)
        for key in reranked_result:
            
            proj, bug_id, path, start_loc, end_loc = key.split('-')

            targetFile = outputDir / (projName + '-' + bug_id + '.txt')
            if targetFile.exists():
                targetFile.unlink(missing_ok=True)

            print('===== Recovering Mutant-{} ====='.format(bug_id))
            
            # bug_start_time = time.time()
            i = 0
            for tokenized_patch in reranked_result[key]['patches']:
                # print('***** Patch-{} *****'.format(i))
                i += 1
                score = tokenized_patch['score']
                tokenized_patch = tokenized_patch['patch']
                # print("tokenized_patch: '{}'".format(str(tokenized_patch)))

                strings, numbers = get_strings_numbers(tmp_dir + path, (int(start_loc) + int(end_loc)) // 2)
                strings = [item[0] for item in strings][:5]
                numbers = [item[0] for item in numbers][:5]
                # print('strings: ' + str(strings))
                # print('numbers: ' + str(numbers))
                # one tokenized patch may be reconstructed to multiple source-code patches
                reconstructed_patches = tokenization.token2statement(tokenized_patch.split(' '), numbers, strings)
                # print('reconstructed_patches: ' + str(reconstructed_patches))
                # validate most 5 source-code patches come from the same tokenized patch
                
                with targetFile.open(mode='a') as t:
                    for patch in reconstructed_patches:  # originally they use reconstructed_patches[:5]
                        patch = patch.strip()
                        t.write(patch + '\n')

# for path in d4jProjPaths:
#     print(str(path))
#     print(getSampledMutIdList(path))
#     print(len(getSampledMutIdList(path)))

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    main()
    # checkCorrectFixes()