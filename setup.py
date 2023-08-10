
from distutils.core import setup

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
  name = 'LRU-tensorflow',         
  packages = ['LRU_tensorflow'],  
  version = '0.1.0',     
  license='MIT',       
  description = 'Unofficial TensorFlow implementation of a Linear Recurrent Unit, proposed by Google Deepmind.',
  long_description=long_description,
  long_description_content_type='text/markdown',  
  author = 'Udit Sharma',                  
  author_email = 'uditsharma.eng@gmail.com',     
  url = 'https://github.com/DemandredEng/Linear-Recurrent-Units-Tensorflow',   
  download_url = '',  
  keywords = ['Artificial Intelligence', 'Deep Learning', 'Recurrent Neural Networks', 'Linear Recurrent Unit'],   
  install_requires=[            
          'tensorflow>=2.0'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',    
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    
  ],
)
