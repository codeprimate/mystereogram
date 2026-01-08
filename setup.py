from setuptools import find_packages, setup


setup(
    name="mystereogram",
    version="0.1.0",
    description="Python CLI autostereogram generator",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0,<3.0.0",
        "torchvision>=0.16.0,<0.18.0",
        "transformers>=4.30.0,<4.41.0",
        "Pillow>=9.0.0,<11.0.0",
        "numpy>=1.21.0,<2.0.0",
        "noise>=1.0.0",
        "rich>=13.0.0,<14.0.0",
        "gradio>=6.0.0",
    ],
    entry_points={
        "console_scripts": [
            "mystereogram=stereogram_generator.cli:main",
            "mystereogram-web=stereogram_generator.web_ui:main",
        ]
    },
    python_requires=">=3.8",
)
