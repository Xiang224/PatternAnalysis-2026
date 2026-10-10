"""Verify the downloaded full ISIC 2017 splits and official YOLO11n checkpoint."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import numpy as np
from PIL import Image

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from modules import load_pretrained_detector

SPLITS = [('train','Training',2000),('val','Validation',150),('test','Test_v2',600)]

def check_pair(job):
    identifier,image,mask=job
    with Image.open(image) as source:
        source.load()
        size=source.size
        mode=source.mode
    with Image.open(mask) as source:
        source.load()
        if source.size != size:
            raise ValueError(f'Image/mask dimensions differ: {identifier}')
        a=np.asarray(source.convert('L'))
        values=np.unique(a).tolist()
        foreground=a>=128
        if not foreground.any():
            raise ValueError(f'Empty mask: {identifier}')
    return {'id':identifier,'size':list(size),'image_mode':mode,'mask_values':values}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root',type=Path,default=Path(__file__).resolve().parents[4])
    args=parser.parse_args()
    root=args.workspace_root.resolve()
    data=root/'data/ISIC2017'
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'splits':{},
            'class_mapping':{'0':'melanoma','1':'nevus','2':'seborrheic_keratosis'},
            'patient_independence_verified':False,'training_run':False}
    id_sets={}
    for split,prefix,expected in SPLITS:
        image_root=data/'raw'/f'ISIC-2017_{prefix}_Data'
        mask_root=data/'raw'/f'ISIC-2017_{prefix}_Part1_GroundTruth'
        images={p.stem:p for p in image_root.glob('*.jpg')}
        masks={p.stem.removesuffix('_segmentation'):p for p in mask_root.glob('*_segmentation.png')}
        label_path=data/'metadata'/f'ISIC-2017_{prefix}_Part3_GroundTruth.csv'
        with label_path.open(newline='',encoding='utf-8-sig') as source:
            rows=list(csv.DictReader(source))
        labels={r['image_id']:r for r in rows}
        if len(labels)!=len(rows): raise ValueError(f'Duplicate labels in {split}')
        if not (len(images)==len(masks)==len(labels)==expected):
            raise ValueError(f'{split}: unexpected counts {len(images)}, {len(masks)}, {len(labels)}; expected {expected}')
        if not (set(images)==set(masks)==set(labels)):
            raise ValueError(f'{split}: image, mask, label IDs do not match')
        counts=Counter()
        for row in rows:
            pair=(float(row['melanoma']),float(row['seborrheic_keratosis']))
            mapping={(1.,0.):'melanoma',(0.,0.):'nevus',(0.,1.):'seborrheic_keratosis'}
            if pair not in mapping: raise ValueError(f'Invalid diagnostic labels: {row}')
            counts[mapping[pair]]+=1
        with ThreadPoolExecutor(max_workers=4) as pool:
            checked=list(pool.map(check_pair,[(i,images[i],masks[i]) for i in sorted(images)]))
        summary={'images':len(images),'masks':len(masks),'diagnoses':len(labels),
                 'class_counts':dict(counts),'all_images_and_masks_decoded':True,
                 'dimension_mismatches':0,'empty_masks':0,
                 'mask_values':sorted({v for r in checked for v in r['mask_values']}),
                 'unique_image_sizes':len({tuple(r['size']) for r in checked}),
                 'image_root':str(image_root),'mask_root':str(mask_root),'labels':str(label_path)}
        report['splits'][split]=summary
        id_sets[split]=set(images)
        print(split,json.dumps(summary),flush=True)
    overlaps={f'{a}_{b}':len(id_sets[a]&id_sets[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    if any(overlaps.values()): raise ValueError(f'Image ID overlap: {overlaps}')
    report['image_id_overlap']=overlaps
    weights=root/'weights/yolo11n.pt'
    digest=hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest!='0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1':
        raise ValueError('Official model SHA256 mismatch')
    model=load_pretrained_detector(weights)
    import torch
    results=model.predict(np.zeros((320,320,3),dtype=np.uint8),imgsz=320,device='cpu',verbose=False,save=False)
    report['model']={'path':str(weights),'sha256':digest,'task':model.task,
                     'pretrained_classes':len(model.names),'cpu_smoke_test':len(results)==1,
                     'smoke_test_input':'synthetic all-zero 320x320 image; not a task performance test'}
    report['environment']={'python':sys.version,'platform':platform.platform(),
          'versions':{p:version(p) for p in ['torch','torchvision','ultralytics','numpy','pillow']},
          'mps_available':torch.backends.mps.is_available(),'cuda_available':torch.cuda.is_available()}
    output=root/'output/setup_verification.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2))
    print('Verification passed:',output,flush=True)
    print(json.dumps(report['model']),flush=True)

if __name__=='__main__': main()
