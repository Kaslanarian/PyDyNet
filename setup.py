import setuptools

setuptools.setup(
    name='pydynet',
    version='0.0.7',
    description=
    'Neuron network(DNN, CNN, RNN, etc) implementation using Numpy based on autodiff',
    author="Welt Xing",
    author_email="xingcy@smail.nju.edu.cn",
    maintainer="Welt Xing",
    maintainer_email="xingcy@smail.nju.edu.cn",
    packages=['pydynet', 'pydynet/nn', 'pydynet/nn/modules'],
    license='MIT License',
    install_requires=['numpy', 'cupy'],
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/Kaslanarian/PyDyNet',
)
