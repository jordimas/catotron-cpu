import argparse
import falcon
import os
from synthesizer import Synthesizer

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
HTML_BODY = os.path.join(PROJECT_PATH, 'templates/demo.html')

class UIResource:
  def on_get(self, req, res):
    res.content_type = 'text/html'
    with open(HTML_BODY, 'r') as html_file:
        res.body = html_file.read()

class SynthesisResource:
  def on_get(self, req, res):
    if not req.params.get('text'):
      raise falcon.HTTPBadRequest()
    res.data = synthesizer.synthesize(req.params.get('text'))
    res.content_type = 'audio/wav'

synthesizer = Synthesizer()
api = falcon.API()
api.add_route('/synthesize', SynthesisResource())
api.add_route('/', UIResource())

if __name__ == '__main__':
  from wsgiref import simple_server
  parser = argparse.ArgumentParser()
  parser.add_argument('--t_checkpoint', help='Full path to tacotron2 checkpoint')
  parser.add_argument('--v_checkpoint', help='Full path to melgan checkpoint')
  parser.add_argument('--port', type=int, default=9000)
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  if args.t_checkpoint and args.v_checkpoint:
      synthesizer.load(args.t_checkpoint, args.v_checkpoint)
  else:
      t_model_path = os.path.join(PROJECT_PATH, 'models/upc_ona_tacotron2.pt')
      v_model_path = os.path.join(PROJECT_PATH, 'models/melgan_LJ11_epoch.pt')
      synthesizer.load(t_model_path, v_model_path)
  print('Serving on port %d' % args.port)
  simple_server.make_server('0.0.0.0', args.port, api).serve_forever()
else:
  synthesizer.load('./models/upc_ona_tacotron2.pt', './models/melgan_LJ11_epoch.pt')
