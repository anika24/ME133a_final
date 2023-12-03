from setuptools import find_packages, setup
from glob import glob

package_name = '133afinal'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*')),
        ('share/' + package_name + '/rviz',   glob('rviz/*')),
        ('share/' + package_name + '/urdf',   glob('urdf/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='aarora@caltech.edu',
    description='ME 133a Final Project: Anika Arora & Andy Dimanku',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'p1_bending = 133afinal.p1_bending:main',
            'p2_dribbling = 133afinal.p2_dribbling:main',
            'p3_shooting = 133afinal.p3_shooting:main'
        ],
    },
)
