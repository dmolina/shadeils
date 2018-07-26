rm shadeils.zip 2>/dev/null
cd ea
rm -Rf build
rm -Rf distr
# python setup.py clean >/dev/null
cd ../..
zip -q -r --exclude=*shadeils-env/* --exclude=*.csv --exclude=*results* --exclude=*__pycache__* --exclude=*out_* shadeils.zip shadeils
cd shadeils
mv ../shadeils.zip .
