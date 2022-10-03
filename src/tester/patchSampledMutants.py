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


TESTER_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
d4jProjDirPath = Path(TESTER_DIR + '../../d4jProj')
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
        print('=' * 10 + " Start {} ".format(projName) + '=' * 10)
        mutIds=getSampledMutIdList(projPath)
        redirectOutErrToLogsAllin(projName)
        prepareMutInputAllIn(projPath, projName, mutIds=mutIds)
        generatePatchesAllIn(projName)
        generateMeta(projPath, projName, allin=True, mutIds=mutIds)
        rerank(projPath, projName, allin=True)
        
# for path in d4jProjPaths:
#     print(str(path))
#     print(getSampledMutIdList(path))
#     print(len(getSampledMutIdList(path)))

if __name__ == '__main__':
    main()