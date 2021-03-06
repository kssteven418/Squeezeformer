# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fr:
    requirements = fr.read().splitlines()

setuptools.setup(
    name="squeezeformer",
    packages=setuptools.find_packages(include=["src*"]),
    install_requires=requirements,
    extras_require={
        #"tf2.3": ["tensorflow>=2.3.0,<2.4", "tensorflow-text>2.3.0,<2.4", "tensorflow-io>=0.16.0,<0.17"],
        #"tf2.3-gpu": ["tensorflow-gpu>=2.3.0,<2.4", "tensorflow-text>=2.3.0,<2.4", "tensorflow-io>=0.16.0,<0.17"],
        #"tf2.4": ["tensorflow>=2.4.0,<2.5", "tensorflow-text>=2.4.0,<2.5", "tensorflow-io>=0.17.0,<0.18"],
        #"tf2.4-gpu": ["tensorflow-gpu>=2.4.0,<2.5", "tensorflow-text>=2.4.0,<2.5", "tensorflow-io>=0.17.0,<0.18"],
        "tf2.5": ["tensorflow>=2.5.0,<2.6", "tensorflow-text>=2.5.0,<2.6", "tensorflow-io>=0.18.0,<0.19"],
        "tf2.5-gpu": ["tensorflow-gpu>=2.5.0,<2.6", "tensorflow-text>=2.5.0,<2.6", "tensorflow-io>=0.18.0,<0.19"]
    },
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    python_requires='>=3.6',
)
