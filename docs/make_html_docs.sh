# script to make the html documentation
make clean
sphinx-apidoc -f -o source/ ../LambdaZero  
make html

# replace static with static everywhere; static is ignored by github.io
cd ./build/html/
mv _static/ static/
for file in `ls | grep .html`; do
    sed 's/_static/static/' $file > tmp
    mv tmp $file
done
cd ../../

# remove all the needless rst files
rm ./source/LambdaZero*.rst
rm ./source/modules.rst
