"""Contact sheet of chosen evidence tiles: sheet_list.py run list.txt out_prefix per_sheet"""
import sys, json, cv2, numpy as np
from pathlib import Path
run=Path(sys.argv[1]); want=[l.split() for l in open(sys.argv[2]) if l.strip()]; pre=sys.argv[3]; per=int(sys.argv[4])
tiles=[]
for cid, stem in want:
    c=json.load(open(run/cid/'case.json'))
    for r in c['exhibit_results']:
        for p in r.get('pages',[]):
            for x in p.get('changes',[]):
                if (x.get('evidence') or '').rsplit('.',1)[0]==stem:
                    im=cv2.imread(str(run/r['evidence_dir']/x['evidence']))
                    s=min(2.5, 900/im.shape[1], 330/im.shape[0]); im=cv2.resize(im,None,fx=s,fy=s,interpolation=cv2.INTER_CUBIC)
                    im=cv2.copyMakeBorder(im,24,6,0,max(0,910-im.shape[1]),cv2.BORDER_CONSTANT,value=(255,255,255))[:, :910]
                    cv2.putText(im,f"{cid} {stem} sim {x['similarity']:.2f} {x['kind']} | {r['exhibit'].get('caption','')[:30]}",(2,17),0,0.55,(0,0,160),1); tiles.append(im)
for k in range(0,len(tiles),per):
    chunk=tiles[k:k+per]; cv2.imwrite(f"{pre}{k//per}.png", np.vstack([cv2.copyMakeBorder(t,0,4,0,0,cv2.BORDER_CONSTANT,value=(200,200,200)) for t in chunk]))
print(len(tiles))
