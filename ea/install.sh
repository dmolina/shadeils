# Create the virtualenv
python -m venv shadeils-env

if [ $? ]; then
    python -m venv shadeils-env --without-pip
    source shadeils-env/bin/activate
    export PATH=$PWD/shadeils-bin/python:$PATH
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python get-pip.py
else
    source shadeils-env/bin/activate
fi
# Install dependencies
pip install cython
pip install -r requirements.txt
# Install locally package
cd ea
python setup.py install
cd ..
