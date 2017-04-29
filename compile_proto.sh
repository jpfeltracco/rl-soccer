#!/bin/bash

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SRC_DIR="$ROOT_DIR/proto"
DST_DIR="$ROOT_DIR/proto/build"
protoc -I=$SRC_DIR --python_out=$DST_DIR $SRC_DIR/*
