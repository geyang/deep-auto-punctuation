author=$(Ge Yang)
author_email=$(yangge1987@gmail.com)

# require python 3.6.0
setup: unzip
run: start-visdom train
unzip:
	mkdir engadget_data
	tar -xvzf engadget_data.tar.gz ./engadget_data
zip:
	tar -czvf engadget_data.tar.gz ./engadget_data/
start-visdom:
	python -m visdom.server &
train:
	bash -c "source activate deep-learning && python -u train.py > train-log.out"


