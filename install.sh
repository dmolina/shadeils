# Create the virtualenv
DIR=venv

if [ ! -d $DIR ]; then
	python -m venv $DIR

	if [ $? ]; then
        echo "Error in previous"
    		python -m venv $DIR --without-pip
    		source $DIR/bin/activate
    		export PATH=$PWD/$DIR/python:$PATH
    		curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    		python get-pip.py
	else
    		source $DIR/bin/activate
	fi

	# Install dependencies
	pip install cython
	pip install -r requirements.txt
	# Install locally package
	cd ea
	python setup.py install
	cd ..
else
	source $DIR/bin/activate
fi

sed -i -E "s?/home/dmolina/shadeils/shadeils-env?${PWD}/${DIR}?" shadeils.py
