import json,sys,os,shutil
src,dst,slug,title=sys.argv[1:5]
edits=json.loads(sys.argv[5]) if len(sys.argv)>5 else []
os.makedirs(dst,exist_ok=True)
m=json.load(open(os.path.join(src,'kernel-metadata.json')))
nbf=m['code_file']; nb=json.load(open(os.path.join(src,nbf)))
n=0
for old,new in edits:
    for c in nb['cells']:
        s=''.join(c['source'])
        if old in s:
            s=s.replace(old,new); c['source']=s; n+=1
assert n==len(edits),(n,edits)
m['id']='kragglenote2forwork/'+slug; m.pop('id_no',None); m['title']=title; m['is_private']=True
m['dataset_sources']=[d for d in m['dataset_sources'] if d]
m['code_file']=slug+'.ipynb'
json.dump(nb,open(os.path.join(dst,slug+'.ipynb'),'w'))
json.dump(m,open(os.path.join(dst,'kernel-metadata.json'),'w'),indent=1)
print('ok',dst,n)
