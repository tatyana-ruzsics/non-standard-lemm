#setup dynet virtual environment
sudo apt-get install python3-venv
python3 -m venv dynet-env
source dynet-env/bin/activate

# CPU:
pip install dynet
# GPU:
#pip install cython
# required newer cmake version than the default of ubuntu 18: https://vitux.com/how-to-install-cmake-on-ubuntu-18-04/
# sudo apt-get install libssl-dev
#BACKEND=cuda pip install git+https://github.com/clab/dynet#egg=dynet

#both:
pip install docopt
pip install progressbar
pip install editdistance
pip install mosestokenizer
pip install matplotlib