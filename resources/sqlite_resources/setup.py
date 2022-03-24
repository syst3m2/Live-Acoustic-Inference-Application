from setuptools import setup

setup(name='sqlite_resources',
      version='0.2',
      description='Helper package which wraps the sqlite3 library, providing simple commands for the sqlite3 databases',
      url='https://gitlab.nps.edu/vector-sensor-processing/python/sqlite_resources',
      author='Paul Leary',
      author_email='pleary@nps.edu',
      license='MIT',
      packages=['sqlite_resources'],
      install_requries=['sqlite3'],
      zip_safe=False)