import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name='graphing_tools',
    version='0.0.2',
    author='Jack Bennetto',
    author_email='jack@bennetto.com',
    description='Assorted graphing functions for data science',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/jackbenn/graphing_tools',
    project_urls = {
        "Bug Tracker": "https://github.com/jackbenn/graphing_tools/issues"
    },
    packages=['graphing_tools'],
    install_requires=['numpy', 'scipy', 'sklearn', 'matplotlib']
    )