# Install chimera on a linux machine

git clone --depth 1 https://github.com/MKorablyov/chimera tmp
cd tmp
cat xaa xab > chimera.bin
chmod 755 chimera.bin
echo '../chimera' | ./chimera.bin
cd ..
rm -rf tmp
cd ..

