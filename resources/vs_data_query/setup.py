from setuptools import setup
from os import path

loc = path.abspath(path.dirname(__file__))

'''
with open(loc + '/requirements.txt') as f:
    contents = f.read().splitlines()

requirements = [
      line
      for line in contents
      if not line.startswith('#')
]
'''
requirements = []

setup(name='vs_data_query',
      version='0.5',
      description='Package which interfaces with the VS data, queries database, and crawls data.',
      url='https://gitlab.nps.edu/vector-sensor-processing/python/vs_data_query',
      author='Paul Leary',
      author_email='pleary@nps.edu',
      license='MIT',
      packages=['vs_data_query'],
      install_requires=requirements,
      zip_safe=False)