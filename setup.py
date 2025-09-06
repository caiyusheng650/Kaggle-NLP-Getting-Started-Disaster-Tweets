from setuptools import setup, find_packages


setup(
    name="nlp-getting-started",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A project for classifying disaster tweets using NLP techniques",
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nlp-getting-started",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "scikit-learn>=1.0.0",
        "transformers>=4.10.0",
        "tokenizers>=0.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
    },
)