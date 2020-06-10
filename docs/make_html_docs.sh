# script to make the html documentation
make clean
sphinx-apidoc -f -o source/ ../LambdaZero  
make html

# remove all the needless rst files
rm ./source/LambdaZero*.rst
rm ./source/modules.rst
