import os
import sys
import subprocess

from synthesizer import Synthesizer

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
synthesizer = Synthesizer()
t_model_path = os.path.join(PROJECT_PATH, 'models/upc_pau_tacotron2.pt')
v_model_path = os.path.join(PROJECT_PATH, 'models/melgan_onapau_catotron.pt')
synthesizer.load(t_model_path, v_model_path)

def main(path):
    files = os.listdir(path)
    files.sort()
    p_wavs = []
    for f in files:
        if f.startswith('para') and f.endswith('.txt'):
            p_file = synthesize_paragraph(os.path.join(path,f))
            p_wavs.append(p_file)
            p_wavs.append('silence_short.wav')

    args = ['sox'] + p_wavs + [os.path.join(path, 'article_full.wav')]
    popen = subprocess.Popen(args)
    popen.wait()

def synthesize_paragraph(f):
    sentences = []
    print(f)
    for i, line in enumerate(open(f).readlines()):
        print(line)

        outfile = f.replace('.txt','_s%i.wav'%i)
        sentences.append(outfile)
        sentences.append('silence_short.wav')
        if not os.path.isfile(outfile):
            audio = synthesizer.synthesize(line)
            with open(outfile, 'wb') as out:
                out.write(audio)

    paragraph_file = f.replace('.txt','.wav')
    if not os.path.isfile(paragraph_file):
        args = ['sox'] + sentences + [paragraph_file]
        popen = subprocess.Popen(args)
        popen.wait()
    return paragraph_file

if __name__ == "__main__":
    path = sys.argv[1]
    main(path)
