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

setup(name='vs_db_maintainer',
      version='0.9',
      description='Package which maintains the database used to index vector sensor audio and orientation data.',
      url='https://gitlab.nps.edu/vector-sensor-processing/python/vs_db_maintainer',
      author='Paul Leary',
      author_email='pleary@nps.edu',
      license='MIT',
      packages=['vs_db_maintainer'],
      install_requires=requirements,
      scripts = [
            'bin/fresh-db',
            'bin/update-db'],
      zip_safe=False)