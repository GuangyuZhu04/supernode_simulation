import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('-o','--output',type=Path,default=ROOT/'outputs/deepseek-v41-fp4-operator-matrix-20261105')
args=parser.parse_args();out=args.output
def load(name):return json.loads((out/name).read_text(encoding='utf-8'))
payload=dict(matrix=load('matrix.compact.json'),presentation=load('presentation_data.json'),profile=load('operator_launch_assumptions.json'))
template=(ROOT/'scripts/templates/v41-fine-matrix-report.html').read_text(encoding='utf-8')
encoded=json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('</','<\\/')
(out/'report.html').write_text(template.replace('__DATA__',encoded),encoding='utf-8')
print(json.dumps(dict(path=str(out/'report.html'),bytes=(out/'report.html').stat().st_size)))
