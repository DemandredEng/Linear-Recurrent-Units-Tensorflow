
from distutils.core import setup
setup(
  name = 'LRU-tensorflow',         
  packages = ['LRU_tensorflow'],  
  version = '0.0.1',     
  license='MIT',       
  description = 'Linear Recurrent Unit (LRU) - TensorFlow',  
  author = 'Udit Sharma',                  
  author_email = 'uditsharma.eng@gmail.com',     
  url = 'https://github.com/DemandredEng/Linear-Recurrent-Units-Tensorflow',   
  download_url = '',  
  keywords = ['Artificial Intelligence', 'Deep Learning', 'Recurrent Neural Networks', 'Linear Recurrent Unit'],   
  install_requires=[            
          'tensorflow>=2.0'
      ],
  classifiers=[
    'Development Status :: 1 - Planning',      
    'Intended Audience :: Developers',    
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    
  ],
)
