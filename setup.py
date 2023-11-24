from setuptools import find_packages, setup
HYPHEN_E_Dot = '-e .'

def get_requirements(filepath):
    requirements = []
    with open(filepath) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if HYPHEN_E_Dot in requirements:
            requirements.remove(HYPHEN_E_Dot)
    return requirements


setup(
    name = 'Customer_Segmentation_Classification',
			    version='0.0.1',
    				author= 'Abhishek',
    				author_email='vedanshtiwari.07@gmail.com',
    				packages=find_packages(),
    				install_requires = get_requirements('requirements.txt')
                    )