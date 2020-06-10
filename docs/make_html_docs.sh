# script to make the html documentation
make clean
sphinx-apidoc -f -o source/ ../LambdaZero  
make html
