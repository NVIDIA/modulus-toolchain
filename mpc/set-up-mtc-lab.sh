python -m venv venv-mtc-lab
./venv-mtc-lab/bin/pip install pip --upgrade
./venv-mtc-lab/bin/pip install -r requirements-mtc-lab.txt

python -m ipykernel install --prefix=./venv-mtc-lab --name 'modulus-python'