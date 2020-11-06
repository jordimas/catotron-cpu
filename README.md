# About Tacotron2_cpu
CPU version of the [NVIDIA Tacotron2](https://github.com/NVIDIA/tacotron2)

## Usage
`python text2speech.py`

## Docker

To build the Docker image:

`docker build -t catotron-cpu . -f Dockerfile`

to run it after it is built:

`docker run -p 8000:8000 -i -t catotron-cpu`

To see it runing open in your browser:

`http://localhost:8000/`

## Thanks

These resources have been developed thanks to the project «síntesi de la parla contra la bretxa digital» (Speech synthesis against the digital gap) that was subsidised by the Department of Culture. A part of the funding comes from the financing administered by the inheritance board of the Generalitat de Catalunya.

Aquests recursos van ser desenvolupats gràcies al projecte «síntesi de la
parla contra la bretxa digital» subvencionat pel Departament de Cultura. Una
part dels fons provenen dels cabals que atorga la Junta d'Herències de la
Generalitat de Catalunya.

<img src="https://github.com/collectivat/cmusphinx-models/raw/master/img/logo_generalitat.png" width="400"/>
