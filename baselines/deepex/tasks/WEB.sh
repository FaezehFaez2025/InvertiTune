export PYTHONPATH=$(pwd)/src:$PYTHONPATH
python scripts/manager.py --task=WEB --cuda=0 --batch-size-per-device=1
