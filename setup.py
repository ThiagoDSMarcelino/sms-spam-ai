from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='sms_spam_ai',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
        ],
    },
    author='Thiago dos Santos Marcelino',
    author_email='thiagodsmarcelino@gmail.com',
    description='A project to classify SMS messages as spam or non-spam using artificial intelligence.',
    url='https://github.com/ThiagoDSMarcelino/sms-spam-ai',
)
