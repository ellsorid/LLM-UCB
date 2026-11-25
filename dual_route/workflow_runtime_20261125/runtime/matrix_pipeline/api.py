"""Bounded provider calls; keys stay in memory and never enter artifacts."""
import sys
sys.dont_write_bytecode=True
import http.client,json,os,time,hashlib,threading,re
from pathlib import Path

def credential(path):
 """Keep the repository's unambiguous DeepSeek text-file convention."""
 data=Path(path).read_bytes();encoding='utf-16' if data.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig'
 lines=data.decode(encoding).splitlines();pattern=re.compile(r'(?<![\w-])sk-[A-Za-z0-9_-]{16,}')
 matches=set(pattern.findall('\n'.join(s for s in lines if 'deepseek' in s.lower())))
 matches=matches or set(pattern.findall('\n'.join(lines)))
 if len(matches)!=1:raise ValueError('Expected one unambiguous DeepSeek credential')
 return matches.pop()

class Client:
 def __init__(self,key_file,output,max_calls=8):
  self._key=credential(Path(key_file)) if key_file else os.environ.get('LOHC_LLM_API_KEY') or os.environ.get('DEEPSEEK_API_KEY')
  if not self._key:raise ValueError('API credential not configured')
  self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True);self.calls=0;self.maximum=max_calls;self.lock=threading.Lock()
  conn=http.client.HTTPSConnection('api.deepseek.com',timeout=45)
  try:
   conn.request('GET','/models',headers={'Authorization':'Bearer '+self._key});r=conn.getresponse()
   if r.status!=200:raise RuntimeError(f'Provider models HTTP {r.status}')
   ids=[x['id'] for x in json.loads(r.read())['data']]
  finally:conn.close()
  self.model=next((m for m in ['deepseek-flash','deepseek-chat'] if m in ids),None)
  if not self.model:raise RuntimeError('No configured supported chat model in provider list')
  (self.output/'provider.json').write_text(json.dumps({'provider':'DeepSeek','model':self.model,'models_check':'HTTP 200','max_completion_requests':max_calls},indent=2),encoding='utf-8')

 def run(self,role,instruction,packet,max_tokens=900):
  from .structure_protocol_adapter import inject
  instruction,packet=inject(role,instruction,packet)
  with self.lock:
   if self.calls>=self.maximum:raise RuntimeError('API request cap reached')
   index=self.calls;self.calls+=1
  payload={'model':self.model,'temperature':0,'max_tokens':max_tokens,'response_format':{'type':'json_object'},
   'messages':[{'role':'system','content':instruction},{'role':'user','content':json.dumps(packet,ensure_ascii=False,separators=(',',':'))}]}
  if self.model=='deepseek-flash':payload['thinking']={'type':'disabled'}
  data=json.dumps(payload,ensure_ascii=False).encode();status=None;response={};start=time.perf_counter()
  conn=http.client.HTTPSConnection('api.deepseek.com',timeout=55)
  try:
   conn.request('POST','/chat/completions',body=data,headers={'Content-Type':'application/json','Authorization':'Bearer '+self._key})
   r=conn.getresponse();status=r.status
   if status!=200:raise RuntimeError(f'Provider completion HTTP {status}')
   response=json.loads(r.read());choice=response['choices'][0]
   if choice.get('finish_reason')=='length':raise ValueError('Truncated provider response')
   content=choice['message']['content'].replace(self._key,'[redacted]');parsed=json.loads(content)
   (self.output/f'{index:02d}_{role}_response.json').write_text(json.dumps(parsed,ensure_ascii=False,indent=2),encoding='utf-8')
   return parsed
  finally:
   conn.close();audit={'role':role,'index':index,'http_status':status,'seconds':time.perf_counter()-start,'model':self.model,
    'request_sha256':hashlib.sha256(data).hexdigest(),'usage':response.get('usage'),'response_id':response.get('id')}
   (self.output/f'{index:02d}_{role}_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
