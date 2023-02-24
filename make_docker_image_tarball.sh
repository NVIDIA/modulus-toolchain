export VERSION=`cat MTC_VERSION`
echo "docker save mtc:$VERSION | gzip > mtc_$VERSION.tar.gz"
echo "Starting on " `date`
time (docker save mtc:$VERSION | gzip > mtc_$VERSION.tar.gz)
echo "Done on " `date`