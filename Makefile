download_and_build: downloaded_v3 all

#DATAFILE=data2.tar.gz
FOLDER = data2

downloaded_v3:
	rm -f downloaded_v2
	find $(FOLDER) -type f -name "*.osg.bz2" -execdir bunzip2 -kf '{}' \;


include $(shell rospack find mk)/cmake.mk
