"""Credential-file gateway transport with bounded, resumable audit records."""
import http.client,json,re,threading,time,hashlib
from pathlib import Path
from urllib.parse import urlsplit

MODELS=['deepseek-v4-pro','qwen3.6-plus','glm-5.2']
CONFIGS=['off_t0','off_t05','thinking']

def parameters(model,config):
 on=config=='thinking';p={}
 if model.startswith('qwen'):p['enable_thinking']=on
 else:p['thinking']={'type':'enabled' if on else 'disabled'}
 if on:
  if model.startswith('deepseek'):p['reasoning_effort']='low'
  if model.startswith('qwen'):p['thinking_budget']=1024
 else:p['temperature']=0 if config=='off_t0' else .5
 return p

class Gateway:
 def __init__(self,key_file,output,limit=144):
  b=Path(key_file).read_bytes();text=b.decode('utf-16' if b.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig')
  urls=re.findall(r'https?://\S+',text);keys=set(re.findall(r'(?<![\w-])sk-[A-Za-z0-9_-]{16,}',text))
  if len(urls)!=1 or len(keys)!=1:raise ValueError('Expected one gateway URL and one credential')
  self.url=urlsplit(urls[0].rstrip('/'));self.key=keys.pop();self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
  self.lock=threading.Lock();self.limit=limit;self.calls=len(list(self.output.glob('*_audit.json')));self.quota_blocked=False
  self.base=self.url.path.rstrip('/')
  if not self.base.endswith('/v1'):self.base+='/v1'
 def clean(self,value):
  return re.sub(r'https?://[^\s\"<>]+','[endpoint]',str(value).replace(self.key,'[redacted]'))
 def request_parameters(self,model,config):return parameters(model,config)
 def call(self,model,config,role,instruction,packet,cap=8192):
  with self.lock:
   if self.quota_blocked:return None,{'status':'not_requested_quota','http_status':None,'model':model,'config':config,'role':role}
   if self.calls>=self.limit:raise RuntimeError('Global request budget exhausted')
   index=self.calls;self.calls+=1
  param=self.request_parameters(model,config)
  payload={'model':model,'messages':[{'role':'system','content':instruction},{'role':'user','content':json.dumps(packet,ensure_ascii=False,separators=(',',':'))}],
   'max_tokens':cap,'response_format':{'type':'json_object'},**param}
  body=json.dumps(payload,ensure_ascii=False).encode();start=time.perf_counter();status=None;result={}
  audit={'index':index,'model':model,'config':config,'role':role,'parameters':param,'max_tokens':cap,'request_sha256':hashlib.sha256(body).hexdigest()}
  conn=(http.client.HTTPSConnection if self.url.scheme=='https' else http.client.HTTPConnection)(self.url.hostname,self.url.port,timeout=getattr(self,'request_timeout',100))
  try:
   conn.request('POST',self.base+'/chat/completions',body=body,headers={'Content-Type':'application/json','Authorization':'Bearer '+self.key})
   r=conn.getresponse();status=r.status;raw=r.read().decode('utf-8')
   if status!=200:
    audit['error_excerpt']=self.clean(raw)[:1000];audit['status']='http_error'
    try:audit['error_code']=json.loads(raw).get('error',{}).get('code')
    except (json.JSONDecodeError,AttributeError):pass
    if audit.get('error_code')=='insufficient_user_quota':self.quota_blocked=True
    return None,audit
   result=json.loads(raw);choice=result['choices'][0];message=choice.get('message',{});content=(message.get('content') or '').replace(self.key,'[redacted]')
   reasoning=message.get('reasoning_content') or message.get('reasoning') or ''
   usage=result.get('usage',{});rt=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens',usage.get('reasoning_tokens'))
   observed=bool(reasoning) or (isinstance(rt,(int,float)) and rt>0)
   audit.update(returned_model=result.get('model'),finish_reason=choice.get('finish_reason'),usage=usage,
    reasoning_tokens=rt,reasoning_characters=len(str(reasoning)),
    thinking_observation='observed' if observed else 'not_reported',
    requested_off_but_reasoning_observed=(config!='thinking' and observed))
   if choice.get('finish_reason')=='length':audit['status']='truncated';return None,audit
   try:parsed=json.loads(content)
   except (json.JSONDecodeError,TypeError):audit['status']='invalid_json';audit['response_excerpt']=content[:1200];return None,audit
   audit['status']='ok'
   (self.output/f'{index:03d}_{role}_response.json').write_text(json.dumps(parsed,ensure_ascii=False,indent=2),encoding='utf-8')
   return parsed,audit
  except Exception as exc:
   audit.update(status='exception',error_type=type(exc).__name__);return None,audit
  finally:
   conn.close();audit.update(http_status=status,seconds=time.perf_counter()-start)
   (self.output/f'{index:03d}_{role}_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')

class ModelClient:
 def __init__(self,transport,model,config,phase):self.transport=transport;self.model=model;self.config=config;self.phase=phase;self.calls=0
 def run(self,role,instruction,packet,max_tokens=1800):
  from .structure_protocol_adapter import inject
  instruction,packet=inject(role,instruction,packet)
  self.calls+=1;parsed,audit=self.transport.call(self.model,self.config,self.phase+'_'+role,instruction,packet,8192)
  if audit['status']!='ok':raise RuntimeError('Gateway '+audit['status'])
  return parsed
