from setuptools import setup

setup(name='rsc',
      version='0.1',
      description='Robust Spectral Clustering for Noisy Data: '
                  'Modeling Sparse Corruptions Improves Latent Embeddings',
      author='Aleksandar Bojchevski, Yves Matkovic, Stephan Günnemann',
      author_email='bojchevs@in.tum.de',
      packages=['rsc'],
      install_requires=['numpy', 'scipy', 'scikit-learn'],
      zip_safe=False)
