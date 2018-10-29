# Mask RCNN Docker Environment

There are supported both, **CPU** and **GPU** versions.
To run **GPU** the **nvidia-docker 2** has to be installed.

## How to

1. **CPU version** run `docker-compose`

    ~~~bash
    cd docker
    docker-compose run --rm --service-ports bash
    ~~~

1. **GPU version** run `docker-compose`

    ~~~bash
    cd docker
    docker-compose run --rm --service-ports bash-gpu
    ~~~
