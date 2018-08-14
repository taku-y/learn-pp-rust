#!/bin/bash
docker run -it --rm \
    --volume $(pwd)/workspace:/root/workspace \
    --name learn-pp-rust learn-pp-rust
