author=$(Ge Yang)
author_email=$(yangge1987@gmail.com)

default:
	make unzip
unzip:
	mkdir engadget_data
	tar -xvzf engadget_data.tar.gz ./engadget_data
zip:
	tar -czvf engadget_data.tar.gz ./engadget_data/
train:
	python -u train.py > train-log.out


