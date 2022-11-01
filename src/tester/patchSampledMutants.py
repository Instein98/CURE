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
d4jProjDirPath = Path(TESTER_DIR + '../../../dataset/d4jProj')
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

def getMutator(projPath: Path, mutId: str):
    mutLog = projPath / 'mutants.log'
    assert mutLog.exists()
    with mutLog.open() as log:
        for line in log:
            if line.startswith(mutId + ':'):
                m = re.match(r'\d+:(\w+):.*\n', line)
                assert m is not None
                return m[1]

def main():
    for projPath in d4jProjPaths:
        projName = projPath.stem

        # if ("math" not in projName):
        #     continue

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

def recoverAndCheckCorrectFixes():
    fixedDict = {}
    mutatorDict = {}
    for dir in mutResultDirPath.iterdir():
        if dir.is_dir():
            resultJson = dir / 'reranked_patches.json'
            if resultJson.exists():
                print('='*10 + dir.stem + '='*10)
                projName = dir.stem.split('-')[0]
                projPathStem = re.match(r'(\w+?-\d+f).*', dir.stem)[1]
                projPath = d4jProjDirPath / projPathStem
                # print(str(projPath))
                # print(projPath.exists())
                recoverPatches(projPath, resultJson, dir / 'patches', projName, candidate_size=100, evenly=False)  # Todo: set candidate_size to 200 or 100?
                
                patchDirPath = dir / 'patches'
                fixedMidList = checkCorrectFix(projPath, patchDirPath)
                fixedDict[projName] = fixedMidList
                for mid in fixedMidList:
                    mutator = getMutator(projPath, mid)
                    if mutator not in mutatorDict:
                        mutatorDict[mutator] = []
                    mutatorDict[mutator].append(projName + '-' + mid)
                # recoverPatches(projPath, resultJson, dir / 'patches-evenly', projName, candidate_size=200, evenly=True)
                # patchDict = json.load()
    proj = [k for k in fixedDict]
    proj.sort()
    for key in proj:
        print("{} mutants of {} are correctly (exactly) fixed!".format(len(fixedDict[key]), key))
    mutators = [k for k in mutatorDict]
    mutators.sort()
    for m in mutators:
        print("{}: {}".format(m, len(mutatorDict[m])))

def checkCorrectFix(projPath: Path, patchDirPath: Path):
    fixedMidList = []
    projSrcRelativePath = sp.check_output("defects4j export -p dir.src.classes", shell=True, universal_newlines=True, cwd=str(projPath), stderr=sp.DEVNULL).strip()
    for patchFile in patchDirPath.iterdir():
        if str(patchFile).endswith('.txt'):
            mid = patchFile.stem.split('-')[1]
            pid = patchFile.stem.split('-')[0]
            fixedLine = getMutFixedLine(projPath, mid, projSrcPath=projSrcRelativePath)
            with patchFile.open() as f:
                for line in f:
                    if isExactlySameCode(line, fixedLine):
                        fixedMidList.append(mid)
                        break
    # print("{} mutants of {} are correctly (exactly) fixed!".format(len(fixedMidList), pid))
    return fixedMidList

def isExactlySameCode(a:str, b:str):
    tmp1 = ''.join(a.split())
    tmp2 = ''.join(b.split())
    return tmp1 == tmp2

def recoverPatches(projPath: Path, rankedJsonPath: Path, outputDir: Path, projName: str, candidate_size=-1, evenly=False, overwrite=False):
    outputDir.mkdir(exist_ok=True, parents=True)
    tmp_dir = str(projPath) + '/'
    reranked_result_path = str(rankedJsonPath)

    with rankedJsonPath.open() as f:
        reranked_result = json.load(f)
        for key in reranked_result:
            
            proj, bug_id, path, start_loc, end_loc = key.split('-')

            targetFile = outputDir / (projName + '-' + bug_id + '.txt')
            # Override mode!
            if targetFile.exists():
                targetFile.unlink(missing_ok=True)

            print('===== Recovering Mutant-{} ====='.format(bug_id))
            
            # bug_start_time = time.time()
            i = 0
            resPatches = []
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
                reconstructed_patches = reconstructed_patches[:5]  # originally they use reconstructed_patches[:5]
                # print('reconstructed_patches: ' + str(reconstructed_patches))
                # validate most 5 source-code patches come from the same tokenized patch

                # force the number of output patches to be candidate_size, candidate_size=-1 means no number limitation
                if candidate_size > 0:
                    # each abstract patch only generate one concrete patch
                    if evenly:
                        resPatches.append(reconstructed_patches[0])
                    # each abstract patch can generate several concrete patch, use top-candi_size of the concrete patches
                    else:
                        if len(resPatches) >= candidate_size:
                            break
                        elif len(resPatches) + len(reconstructed_patches) <= candidate_size:
                            resPatches.extend(reconstructed_patches)
                        else:
                            candLeft = candidate_size - len(resPatches)
                            resPatches.extend(reconstructed_patches[:candLeft])
                            break
                else:
                    resPatches.extend(reconstructed_patches)
            with targetFile.open(mode='a') as t:
                for patch in resPatches:
                    patch = patch.strip()
                    t.write(patch + '\n')

def getMutLineNum(projPath: Path, mutId: str):
    mutLog = projPath / 'mutants.log'
    assert mutLog.exists()
    with mutLog.open() as log:
        for line in log:
            if line.startswith(mutId + ':'):
                m = re.match(r'.+:(\d+):.+\n', line)
                if (m is None):
                    print("Mutant-{} has no match for '.+:(\d+):[^:]+' in line {}".format(mutId, line))
                assert m is not None
                return int(m[1])

def doGeneratePatchedJavaFile(originalFile: Path, repalceLineNum: int, replaceContent: str, outputFile: Path):
    with originalFile.open() as f:
        contents = f.readlines()
    contents[repalceLineNum-1] = replaceContent
    outputFile.parent.mkdir(parents=True, exist_ok=True)
    with outputFile.open(mode='w') as f:
        for line in contents:
            f.write(line)

def generatePatchedJavaFile(projPath: Path, mid: str, patchLine: str, outputDir: Path, projSrcPath=None):
    projSrcRelativePath = sp.check_output("defects4j export -p dir.src.classes", shell=True, universal_newlines=True, cwd=str(projPath), stderr=sp.DEVNULL).strip() if projSrcPath is None else projSrcPath
    shortPath = sp.check_output('find . -name "*.java"', shell=True, universal_newlines=True, cwd=str(projPath / 'mutants' / mid)).strip()
    mutLineNum = getMutLineNum(projPath, mid)
    originalJavaFilePath = projPath / 'mutants' / mid / shortPath
    doGeneratePatchedJavaFile(originalJavaFilePath, mutLineNum, patchLine, outputDir / shortPath)

patchedSourceOutputDir = Path('cure_patches').resolve()

def collectSourcePatches():
    for resultDir in mutResultDirPath.iterdir():
        if not resultDir.is_dir():
            continue
        projName = resultDir.name.split('-')[0]
        for projPath in d4jProjPaths:
            if projName == projPath.stem.split('-')[0]:
                projSrcRelativePath = sp.check_output("defects4j export -p dir.src.classes", shell=True, universal_newlines=True, cwd=str(projPath), stderr=sp.DEVNULL).strip()
                patchDir = resultDir / 'patches'
                assert patchDir.exists()
                for patchFile in patchDir.iterdir():
                    name, mid = patchFile.stem.split('-')
                    assert name == projName
                    with patchFile.open() as f:
                        patchId = 1
                        for patchLine in f:
                            print('Processing {}-{}-{}'.format(projName, mid, patchId))
                            targetOutputDir = patchedSourceOutputDir / (projName + '_' + mid) / 'patches-pool' / str(patchId)
                            generatePatchedJavaFile(projPath, mid, patchLine, targetOutputDir, projSrcPath=projSrcRelativePath)
                            patchId += 1


# for path in d4jProjPaths:
#     print(str(path))
#     print(getSampledMutIdList(path))
#     print(len(getSampledMutIdList(path)))

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # main()
    # recoverAndCheckCorrectFixes()
    collectSourcePatches()