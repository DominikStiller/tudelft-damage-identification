#!/usr/bin/env bash

if [ -d "venv" ]
then
  source venv/bin/activate
fi

python -m unittest -v -b $@
