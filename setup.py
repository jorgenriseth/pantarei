# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['pantarei', 'pantarei.io']

package_data = \
{'': ['*']}

install_requires = \
['fenics>=2019.1.0,<2020.0.0',
 'jupyter>=1.0.0,<2.0.0',
 'matplotlib>=3.5.1,<4.0.0']

setup_kwargs = {
    'name': 'pantarei',
    'version': '0.1.0',
    'description': '',
    'long_description': None,
    'author': 'jorgenriseth',
    'author_email': 'jnriseth@gmail.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.9,<4.0',
}


setup(**setup_kwargs)

