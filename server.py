import argparse
import falcon
import os
from synthesizer import Synthesizer

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
HTML_BODY = os.path.join(PROJECT_PATH, 'templates/demo.html')

class UIResource:
  def on_get(self, req, res):
    res.content_type = 'text/html'
    with open(HTML_BODY, 'r', encoding='utf-8') as html_file:
        res.body = html_file.read()

class SynthesisResource:
  def on_get(self, req, res):
    if not req.params.get('text'):
      raise falcon.HTTPBadRequest()
    if len(req.params.get('text')) > 150:
      raise falcon.HTTPBadRequest('String too long',
                                  'String length shorter than 150 is accepted.')
    res.data = synthesizer.synthesize(req.params.get('text'))
    res.content_type = 'audio/wav'

synthesizer = Synthesizer()
# TODO load via config
t_model_path = os.path.join(PROJECT_PATH, 'models/upc_ona_tacotron2.pt')
v_model_path = os.path.join(PROJECT_PATH, 'models/melgan_onapau_catotron.pt')
synthesizer.load(t_model_path, v_model_path)

app = falcon.API()
app.add_route('/synthesize', SynthesisResource())
app.add_route('/', UIResource())

if __name__ == '__main__':
    from wsgiref import simple_server
    port = 9000
    print('Serving on port %d' % port)
    simple_server.make_server('0.0.0.0', port, app).serve_forever()
