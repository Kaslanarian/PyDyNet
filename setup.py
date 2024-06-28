import setuptools

setuptools.setup(
    name='pydynet',
    version='0.1',
    description=
    'PyDyNet: Neuron Network (MLP, CNN, RNN, Transformer, ...) implementation using Numpy with Autodiff',
    author="Welt Xing",
    author_email="xingcy@smail.nju.edu.cn",
    maintainer="Welt Xing",
    maintainer_email="xingcy@smail.nju.edu.cn",
    packages=['pydynet', 'pydynet/optim', 'pydynet/nn', 'pydynet/nn/modules'],
    license='MIT License',
    install_requires=['numpy'],
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/Kaslanarian/PyDyNet',
)
