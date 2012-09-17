download_and_build: downloaded_v2 all

DATAFILE=data2.tar.gz
FOLDER = data2

downloaded_v2:
	rm -rf $(FOLDER)
	rm downloaded_v1
	tar xvf $(DATAFILE)	
	touch downloaded_v2


include $(shell rospack find mk)/cmake.mk
