download_and_build: downloaded_v1 all

DATAFILE=data2.tar.gz
FOLDER = data2

downloaded_v1:
	rm -rf $(FOLDER)
	tar xvf $(DATAFILE)	
	touch downloaded_v1


include $(shell rospack find mk)/cmake.mk
