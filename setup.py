import setuptools

with open("README.md", "r",encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="mcvqoe-psud",
    author="Jesse Frey, Peter Fink, Jaden Pieper",
    author_email="jesse.frey@nist.gov,jaden.pieper@nist.gov",
    description="Measurement code for the probability of successful delivery (PSuD)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.nist.gov/gitlab/PSCR/MCV/psud",
    packages=setuptools.find_namespace_packages(include=['mcvqoe.*']),
    include_package_data=True,
    use_scm_version={'write_to' : 'mcvqoe/psud/version.py'},
    setup_requires=['setuptools_scm'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Public Domain",
        "Operating System :: OS Independent",
    ],
    license='NIST software License',
    install_requires=[
        'mcvqoe-nist',
        'abcmrt-nist',
    ],
    entry_points={
        'console_scripts':[
            'psud-sim=mcvqoe.psud.PSuD_simulate:main',
            'psud-measure-1loc=mcvqoe.psud.PSuD_1way_1loc:main',
            'psud-eval=mcvqoe.psud.PSuD_eval:main',
            'psud-reprocess=mcvqoe.psud.PSuD_reprocess:main',
        ],
    },
    python_requires='>=3.6',
)

