from .runtime import main
import sys,json,traceback
from pathlib import Path
try:
 main()
except Exception as exc:
 if '--output' in sys.argv:
  output=Path(sys.argv[sys.argv.index('--output')+1])
  if output.is_dir():
   (output/'failure.json').write_text(json.dumps({'status':'failed','exception_type':type(exc).__name__,
    'stack':[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(exc.__traceback__)]},indent=2),encoding='utf-8')
 raise
